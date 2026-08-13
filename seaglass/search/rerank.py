"""`search/rerank.py` — cross-encoder reranking via a hand-loaded
`mlx-embeddings` BERT encoder plus a manually-attached classifier head.

**Why this module exists instead of just calling `mlx_embeddings.load()`:**
PLAN.md §4/§6 assumes `mlx-embeddings` loads `cross-encoder/ms-marco-MiniLM-
L-12-v2` directly ("mlx-embeddings # embeddings AND cross-encoder
reranking"). That assumption does NOT hold as of `mlx-embeddings` 0.1.0:
`load()` raises `ValueError: Received 201 parameters not in model` because
the checkpoint's weights are namespaced `bert.*` + a separate
`classifier.{weight,bias}` (HF `BertForSequenceClassification` layout),
while `mlx_embeddings.models.bert.Model` expects unprefixed encoder/pooler
weights and has no classifier head at all -- it's built for plain
embedding models like `bge-small-en-v1.5`, not for sequence-classification
cross-encoders.

The fix (validated in a spike, see ADDENDUM.md): the encoder architecture
underneath is identical. Strip the `bert.` prefix, load the encoder+pooler
into `mlx_embeddings.models.bert.Model` normally, then apply the
classifier's weight/bias (a single `Linear(384, 1)`, extracted straight
from the checkpoint's safetensors) to the pooler output ourselves. Spiked
latency: ~22ms warm for a batch of 50 (query, candidate) pairs -- inside
PLAN.md's ~60ms reranker budget.

This is intentionally a small, self-contained loader rather than a
generic "load any HF sequence-classification model in MLX" utility --
it's scoped to the one reranker checkpoint this system uses.
"""

from __future__ import annotations

import dataclasses
import json
from typing import List, Sequence, Tuple

RERANK_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-12-v2"
MAX_SEQ_LENGTH = 512


@dataclasses.dataclass
class CrossEncoderReranker:
    """Lazy-loading cross-encoder reranker. Loading requires network access
    on first run (HF Hub download, cached thereafter) -- kept out of the
    constructor so importing this module never touches the network.
    """

    model_id: str = RERANK_MODEL_ID
    _model: object = dataclasses.field(default=None, repr=False, init=False)
    _tokenizer: object = dataclasses.field(default=None, repr=False, init=False)
    _classifier_w: object = dataclasses.field(default=None, repr=False, init=False)
    _classifier_b: object = dataclasses.field(default=None, repr=False, init=False)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import mlx.core as mx
        from mlx_embeddings.models.bert import Model, ModelArgs

        from seaglass.mlxmem import configure_mlx_memory

        configure_mlx_memory()
        from mlx_embeddings.utils import get_model_path, load_tokenizer
        from safetensors import safe_open

        path = get_model_path(self.model_id)
        with open(path / "config.json") as config_file:
            config = json.load(config_file)
        config["model_type"] = "bert"
        arg_fields = set(ModelArgs.__dataclass_fields__)
        model_args = ModelArgs(**{k: v for k, v in config.items() if k in arg_fields})
        model = Model(model_args)

        encoder_weights = []
        classifier_w = classifier_b = None
        with safe_open(path / "model.safetensors", framework="numpy") as reader:
            for key in reader.keys():
                if key == "bert.embeddings.position_ids":
                    continue  # a registered buffer, not a learned param
                array = mx.array(reader.get_tensor(key))
                if key.startswith("bert."):
                    encoder_weights.append((key[len("bert."):], array))
                elif key == "classifier.weight":
                    classifier_w = array
                elif key == "classifier.bias":
                    classifier_b = array
        if classifier_w is None or classifier_b is None:
            raise RuntimeError(
                f"{self.model_id}: expected top-level 'classifier.weight'/'classifier.bias' "
                "tensors (HF BertForSequenceClassification layout) -- checkpoint layout "
                "may have changed; update this loader."
            )
        model.load_weights(encoder_weights, strict=True)
        model.eval()  # BUG FIX: mlx_embeddings.utils.load_model() calls this for
        # you when using load(); our hand-rolled loader must call it too, or
        # nn.Dropout (hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
        # stays active at inference, making score() nondeterministic run-to-run
        # (measured std ~1.8 logits / rank-order flips across identical calls).

        self._model = model
        self._tokenizer = load_tokenizer(path)
        self._classifier_w = classifier_w
        self._classifier_b = classifier_b

    def score(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """Score (query, candidate_text) pairs, batched. Higher = more
        relevant; scores are raw logits (the checkpoint's activation
        function is Identity -- see its config.json), not probabilities,
        so only relative order/magnitude across a batch is meaningful.
        """
        self._ensure_loaded()
        import mlx.core as mx

        batch = self._tokenizer(
            [[q, c] for q, c in pairs],
            return_tensors="mlx",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
        out = self._model(
            batch["input_ids"],
            token_type_ids=batch.get("token_type_ids"),
            attention_mask=batch.get("attention_mask"),
        )
        # BUG FIX: token_type_ids were previously never passed, so
        # BertEmbeddings substituted an all-zero segment id for every
        # token -- off-distribution input for a [CLS] query [SEP] doc [SEP]
        # cross-encoder trained with real segment embeddings. Passing the
        # tokenizer's real token_type_ids widens the relevant/irrelevant
        # margin measurably (verified: true positive +1.64 -> +3.23).
        logits = out.pooler_output @ self._classifier_w.T + self._classifier_b
        mx.eval(logits)
        return [float(x) for x in logits.reshape(-1).tolist()]
