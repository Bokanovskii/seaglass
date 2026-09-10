import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_copilot_plugin_exposes_the_imessage_skill():
    manifest = json.loads((ROOT / "plugin.json").read_text(encoding="utf8"))
    assert manifest["name"] == "seaglass"
    assert manifest["$schema"].endswith("/plugin.schema.json")
    skill = ROOT / "skills" / "imessage" / "SKILL.md"
    assert skill.is_file()
    text = skill.read_text(encoding="utf8")
    assert "name: imessage" in text
    assert "`wait=true`" in text
    assert "Seaglass is read-only" in text

    mcp = json.loads((ROOT / "mcp.json").read_text(encoding="utf8"))
    server = mcp["mcpServers"]["seaglass"]
    assert server["type"] == "stdio"
    assert server["command"] == "./bin/seaglass-mcp"
    assert (ROOT / "bin" / "seaglass-mcp").is_file()
    assert os.access(ROOT / "bin" / "seaglass-mcp", os.X_OK)
