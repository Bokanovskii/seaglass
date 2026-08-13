

class TestSenderFiltering:
    def _session(self):
        from seaglass.search.hydrate import HydratedMessage, HydratedSession

        def message(mid, handle):
            return HydratedMessage(
                message_id=mid, ts=float(mid), is_from_me=handle is None,
                sender=handle, text=f'm{mid}', has_attachment=False, handle=handle,
            )

        return HydratedSession(
            chat_id=1, day='2026-08-12', score=1.0,
            hit_messages=[message(1, '+1555'), message(2, '+1666'), message(3, None)],
            context_messages=[message(4, '+1666')],
        )

    def test_only_the_named_senders_messages_stay_hits(self):
        from seaglass.app.engine import _filter_by_sender

        kept = _filter_by_sender([self._session()], ['+1555'])
        assert [m.message_id for m in kept[0].hit_messages] == [1]

    def test_the_others_are_demoted_to_context_not_dropped(self):
        from seaglass.app.engine import _filter_by_sender

        kept = _filter_by_sender([self._session()], ['+1555'])
        assert {m.message_id for m in kept[0].context_messages} == {2, 3, 4}

    def test_a_session_the_sender_never_spoke_in_is_dropped(self):
        from seaglass.app.engine import _filter_by_sender

        assert _filter_by_sender([self._session()], ['+1999']) == []

    def test_handle_matching_ignores_case(self):
        from seaglass.app.engine import _filter_by_sender

        from dataclasses import replace

        session = self._session()
        session.hit_messages[0] = replace(session.hit_messages[0], handle='Kaya@Example.com')
        kept = _filter_by_sender([session], ['kaya@example.com'])
        assert [m.message_id for m in kept[0].hit_messages] == [1]


class TestBrowseOrdering:
    """Browsing means newest-first, and must not be decided by chunk count."""

    def _rows(self):
        # chat 1 has two chunks on an older day; chat 2 has one newer chunk.
        day = 86400.0
        return [(11, 1, 1_000_000.0), (12, 1, 1_000_000.0 + 60), (21, 2, 1_000_000.0 + 5 * day)]

    def test_a_newer_single_chunk_day_outranks_an_older_busy_one(self):
        from seaglass.search.rank import RankedChunk, order_sessions

        rows = self._rows()
        chunks = [
            RankedChunk(chunk_id=cid, chat_id=chat, start_ts=ts, rerank_score=float(len(rows) - i))
            for i, (cid, chat, ts) in enumerate(rows)
        ]
        ordered = order_sessions(chunks, head_size=5, recent_slots=0)
        newest = {cid: ts for cid, _chat, ts in rows}
        ordered.sort(
            key=lambda s: max((newest.get(c, 0.0) for c in s.hit_chunk_ids), default=0.0),
            reverse=True,
        )
        assert ordered[0].chat_id == 2

    def test_the_pseudo_scores_alone_would_have_got_this_wrong(self):
        # Documents *why* the explicit sort exists: the sigmoid saturates,
        # so summing it ranks by how many chunks a day has.
        from seaglass.search.rank import RankedChunk, order_sessions

        rows = self._rows()
        chunks = [
            RankedChunk(chunk_id=cid, chat_id=chat, start_ts=ts, rerank_score=float(len(rows) - i))
            for i, (cid, chat, ts) in enumerate(rows)
        ]
        assert order_sessions(chunks, head_size=5, recent_slots=0)[0].chat_id == 1


class TestSenderFilterExcludesMyOwnMessages:
    def test_outgoing_messages_stamped_with_the_recipients_handle_are_not_hits(self):
        # chat.db puts the *other* party's handle_id on outgoing 1:1
        # messages, so handle alone matches both halves of the exchange.
        from seaglass.app.engine import _filter_by_sender
        from seaglass.search.hydrate import HydratedMessage, HydratedSession

        def message(mid, from_me):
            return HydratedMessage(
                message_id=mid, ts=float(mid), is_from_me=from_me, sender=None,
                text='x', has_attachment=False, handle='+1555',
            )

        session = HydratedSession(
            chat_id=1, day='d', score=1.0,
            hit_messages=[message(1, False), message(2, True)], context_messages=[],
        )
        kept = _filter_by_sender([session], ['+1555'])
        assert [m.message_id for m in kept[0].hit_messages] == [1]


class TestBrowseKeepsTheNewestHits:
    def _session(self, n):
        from seaglass.search.hydrate import HydratedMessage, HydratedSession

        return HydratedSession(
            chat_id=1, day='d', score=1.0,
            hit_messages=[
                HydratedMessage(message_id=i, ts=float(i), is_from_me=False, sender='A',
                                text=f'm{i}', has_attachment=False, handle='+1555')
                for i in range(n)
            ],
            context_messages=[],
        )

    def test_a_long_session_is_trimmed_to_its_newest_hits(self):
        from seaglass.app.engine import BROWSE_HITS_PER_SESSION, _newest_hits_only

        kept = _newest_hits_only([self._session(30)])[0]
        assert [m.message_id for m in kept.hit_messages] == list(
            range(30 - BROWSE_HITS_PER_SESSION, 30)
        )

    def test_hits_stay_in_reading_order(self):
        from seaglass.app.engine import _newest_hits_only

        kept = _newest_hits_only([self._session(30)])[0]
        assert kept.hit_messages == sorted(kept.hit_messages, key=lambda m: m.ts)

    def test_the_trimmed_messages_become_context_not_losses(self):
        from seaglass.app.engine import _newest_hits_only

        kept = _newest_hits_only([self._session(30)])[0]
        assert len(kept.hit_messages) + len(kept.context_messages) == 30

    def test_a_short_session_is_untouched(self):
        from seaglass.app.engine import _newest_hits_only

        session = self._session(4)
        assert _newest_hits_only([session])[0] is session
