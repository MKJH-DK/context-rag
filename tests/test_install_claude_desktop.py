import json
import sys
from pathlib import Path

from context_rag import cli


def test_install_claude_desktop_creates_fresh_config(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config_path = tmp_path / "claude_desktop_config.json"
    cwd = tmp_path / "corpus"
    cwd.mkdir()
    monkeypatch.setattr(cli, "resolve_claude_desktop_config_path", lambda: config_path)

    assert cli.main(["install-claude-desktop", "--cwd", str(cwd)]) == 0

    config = json.loads(config_path.read_text(encoding="utf-8"))
    entry = config["mcpServers"]["context-rag"]
    assert entry == {
        "command": sys.executable,
        "args": ["-m", "context_rag.cli", "serve"],
        "cwd": str(cwd.resolve()),
        "env": {"PYTHONPATH": str(Path(cli.__file__).resolve().parents[1])},
    }
    assert (
        f'Wrote MCP entry "context-rag" to {config_path}. '
        "Restart Claude Desktop to load."
    ) in capsys.readouterr().out


def test_install_claude_desktop_preserves_existing_servers(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "claude_desktop_config.json"
    cwd = tmp_path / "corpus"
    cwd.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "theme": "dark",
                "mcpServers": {
                    "other": {
                        "command": "node",
                        "args": ["server.js"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "resolve_claude_desktop_config_path", lambda: config_path)

    assert cli.main(["install-claude-desktop", "--name", "rag", "--cwd", str(cwd)]) == 0

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["theme"] == "dark"
    assert config["mcpServers"]["other"]["command"] == "node"
    assert config["mcpServers"]["rag"]["cwd"] == str(cwd.resolve())


def test_install_claude_desktop_refuses_existing_without_force(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config_path = tmp_path / "claude_desktop_config.json"
    config_path.write_text(
        json.dumps({"mcpServers": {"context-rag": {"command": "old"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "resolve_claude_desktop_config_path", lambda: config_path)

    assert cli.main(["install-claude-desktop"]) == 1

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["mcpServers"]["context-rag"]["command"] == "old"
    assert "Re-run with --force to overwrite." in capsys.readouterr().err


def test_install_claude_desktop_force_overwrites(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "claude_desktop_config.json"
    cwd = tmp_path / "corpus"
    cwd.mkdir()
    config_path.write_text(
        json.dumps({"mcpServers": {"context-rag": {"command": "old"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "resolve_claude_desktop_config_path", lambda: config_path)

    assert cli.main(["install-claude-desktop", "--cwd", str(cwd), "--force"]) == 0

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["mcpServers"]["context-rag"]["command"] == sys.executable
    assert config["mcpServers"]["context-rag"]["cwd"] == str(cwd.resolve())


def test_install_claude_desktop_dry_run_does_not_write(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config_path = tmp_path / "claude_desktop_config.json"
    cwd = tmp_path / "corpus"
    cwd.mkdir()
    monkeypatch.setattr(cli, "resolve_claude_desktop_config_path", lambda: config_path)

    assert cli.main(["install-claude-desktop", "--cwd", str(cwd), "--dry-run"]) == 0

    assert not config_path.exists()
    config = json.loads(capsys.readouterr().out)
    assert config["mcpServers"]["context-rag"]["cwd"] == str(cwd.resolve())
