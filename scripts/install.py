#!/usr/bin/env python3
"""Unified dependency installer for context-rag."""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path


class InstallError(RuntimeError):
    pass


def is_termux() -> bool:
    prefix = os.environ.get("PREFIX", "")
    return bool(os.environ.get("TERMUX_VERSION")) or "com.termux" in prefix


def detect_os() -> str:
    if is_termux():
        return "termux"
    system = platform.system().lower()
    if system == "windows":
        return "windows"
    if system == "darwin":
        return "macos"
    if system == "linux":
        return "linux"
    return system or "unknown"


def has_dev_extra(project_root: Path | None = None) -> bool:
    root = project_root or Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    optional = data.get("project", {}).get("optional-dependencies", {})
    return "dev" in optional


def build_pip_command(*, dev: bool, project_root: Path | None = None) -> list[str]:
    editable_target = "."
    if dev and has_dev_extra(project_root):
        editable_target = ".[dev]"
    return [sys.executable, "-m", "pip", "install", "-e", editable_target]


def build_verify_command() -> list[str]:
    return [sys.executable, "-m", "context_rag.cli", "--help"]


def format_command(command: list[str]) -> str:
    return shlex.join(command)


def run_command(command: list[str], *, dry_run: bool, verbose: bool) -> subprocess.CompletedProcess[str] | None:
    print(f"$ {format_command(command)}")
    if dry_run:
        return None
    result = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=not verbose,
    )
    if result.returncode == 0:
        return result
    if not verbose:
        if result.stdout:
            print(result.stdout, file=sys.stderr, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
    raise InstallError(f"Command failed with exit code {result.returncode}: {format_command(command)}")


def verify_cli(*, dry_run: bool) -> tuple[bool, str]:
    if dry_run:
        return False, "not checked in dry-run"
    result = subprocess.run(build_verify_command(), check=False, text=True, capture_output=True)
    detail = (result.stdout or result.stderr or "").splitlines()
    hint = detail[0] if detail else "no output"
    return result.returncode == 0, hint


def print_status_table(rows: list[tuple[str, str, str]]) -> None:
    print("\nFinal status:")
    print(f"{'Target':<18} {'Status':<8} Hint")
    print(f"{'-' * 18} {'-' * 8} {'-' * 40}")
    for target, status, hint in rows:
        print(f"{target:<18} {status:<8} {hint}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install context-rag dependencies.")
    parser.add_argument("--dev", action="store_true", help="install dev extras when available")
    parser.add_argument("--dry-run", action="store_true", help="print planned commands without running them")
    parser.add_argument("--verbose", action="store_true", help="stream full command output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os_name = detect_os()
    print(f"Detected OS: {os_name}")
    print("System binaries: none required")
    if args.dry_run:
        print("Mode: dry-run")

    ok = True
    try:
        run_command(build_pip_command(dev=args.dev), dry_run=args.dry_run, verbose=args.verbose)
    except InstallError as exc:
        print(str(exc), file=sys.stderr)
        ok = False

    cli_ok, detail = verify_cli(dry_run=args.dry_run)
    rows = [("context-rag CLI", "OK" if cli_ok else "MISSING", detail)]
    print_status_table(rows)

    if args.dry_run:
        return 0 if ok else 1
    return 0 if ok and cli_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
