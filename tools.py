"""Three read-only tools over one directory.

Each tool is here because it has a failure mode worth handling: a missing path,
a path that escapes the sandbox, a file too large to read, a regex that does
not compile, a binary file with a .py extension. Those failures are raised as
ToolError subclasses and turned into tool results by the registry -- never into
crashes.
"""

from __future__ import annotations

import re
from pathlib import Path

from agent import Tool, ToolError, ToolRegistry

MAX_FILE_BYTES = 200_000
MAX_GREP_MATCHES = 50
MAX_SCAN_FILES = 2_000
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache", ".mypy_cache"}


class NotFound(ToolError):
    kind = "not_found"


class PermissionDenied(ToolError):
    kind = "permission_denied"


class InvalidArgument(ToolError):
    kind = "invalid_argument"


class Workspace:
    """Confines every tool to a single root directory.

    The sandbox check is the reason path handling is not inline in each tool:
    one place to get it wrong, one place to test it.
    """

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()

    def resolve(self, raw: str) -> Path:
        target = (self.root / raw).resolve()
        if target != self.root and self.root not in target.parents:
            raise InvalidArgument(
                f"path {raw!r} resolves outside the workspace",
                hint=f"every path must stay under {self.root}",
            )
        return target

    def list_dir(self, path: str = ".") -> str:
        target = self.resolve(path)
        if not target.exists():
            raise NotFound(f"no such directory: {path}", hint="call list_dir on '.' to see the layout")
        if not target.is_dir():
            raise InvalidArgument(f"{path!r} is a file, not a directory", hint="use read_file for files")
        try:
            entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError as exc:
            raise PermissionDenied(f"cannot list {path}: {exc.strerror}") from exc

        if not entries:
            return f"{path}: (empty directory)"

        lines = [f"{path}:"]
        for entry in entries:
            if entry.is_dir():
                lines.append(f"  {entry.name}/")
            else:
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = -1
                lines.append(f"  {entry.name}  ({size} bytes)")
        return "\n".join(lines)

    def read_file(self, path: str, start_line: int = 1, max_lines: int = 200) -> str:
        target = self.resolve(path)
        if not target.exists():
            parent = str(Path(path).parent)
            raise NotFound(
                f"no such file: {path}",
                hint=f"call list_dir on {parent!r} to see what actually exists there",
            )
        if target.is_dir():
            raise InvalidArgument(f"{path!r} is a directory", hint="use list_dir for directories")
        try:
            raw = target.read_bytes()
        except PermissionError as exc:
            raise PermissionDenied(f"cannot read {path}: {exc.strerror}") from exc

        cut_bytes = len(raw) > MAX_FILE_BYTES
        if cut_bytes:
            raw = raw[:MAX_FILE_BYTES]
        if b"\x00" in raw[:4096]:
            raise InvalidArgument(
                f"{path!r} contains NUL bytes and looks binary",
                hint="read_file only handles text; grep may still be useful on it",
            )

        lines = raw.decode("utf-8", errors="replace").splitlines()
        start = max(start_line, 1)
        if start > len(lines):
            raise InvalidArgument(
                f"{path!r} has only {len(lines)} lines, so start_line={start_line} is past the end",
                hint=f"start_line must be between 1 and {max(len(lines), 1)}",
            )
        window = lines[start - 1 : start - 1 + max_lines]
        end = start + len(window) - 1

        width = len(str(end))
        body = "\n".join(f"{n:>{width}} | {text}" for n, text in enumerate(window, start))
        header = f"{path} (lines {start}-{end} of {len(lines)})"

        notes = []
        if end < len(lines):
            notes.append(f"{len(lines) - end} more line(s) not shown; raise max_lines or start_line={end + 1}")
        if cut_bytes:
            notes.append(f"file is larger than {MAX_FILE_BYTES} bytes and was cut off mid-way")
        if notes:
            body += "\n\n[truncated] " + " ".join(notes)
        return f"{header}\n{body}"

    def grep(self, pattern: str, path: str = ".", glob: str = "*") -> str:
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            raise InvalidArgument(
                f"invalid regular expression {pattern!r}: {exc}",
                hint="the pattern uses Python re syntax; escape literal brackets and parentheses",
            ) from exc

        base = self.resolve(path)
        if not base.exists():
            raise NotFound(f"no such path: {path}", hint="call list_dir on '.' to see the layout")

        if base.is_file():
            candidates = [base]
        else:
            candidates = []
            for candidate in base.rglob(glob):
                if len(candidates) >= MAX_SCAN_FILES:
                    break
                if candidate.is_file() and not SKIP_DIRS.intersection(candidate.parts):
                    candidates.append(candidate)

        matches: list[str] = []
        scanned = 0
        for candidate in candidates:
            try:
                if candidate.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                continue
            if "\x00" in text[:4096]:
                continue
            scanned += 1
            for number, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    if len(matches) >= MAX_GREP_MATCHES:
                        matches.append(f"... stopped at {MAX_GREP_MATCHES} matches; narrow the pattern or path")
                        return f"{len(matches)} match(es) for {pattern!r} in {path}:\n" + "\n".join(matches)
                    try:
                        shown = candidate.relative_to(self.root)
                    except ValueError:
                        shown = candidate
                    matches.append(f"{shown}:{number}: {line.strip()[:160]}")

        if not matches:
            return f"no matches for {pattern!r} in {path} (scanned {scanned} file(s))"
        return f"{len(matches)} match(es) for {pattern!r} in {path}:\n" + "\n".join(matches)


def build_registry(root: str | Path = ".") -> ToolRegistry:
    """Wire the three tools into a registry rooted at `root`."""
    workspace = Workspace(root)
    return (
        ToolRegistry()
        .add(Tool(
            name="list_dir",
            description="List the entries of a directory. Use this before reading a file whose path you are unsure of.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory relative to the workspace root. Defaults to '.'"},
                },
                "required": [],
            },
            fn=workspace.list_dir,
        ))
        .add(Tool(
            name="read_file",
            description="Read a UTF-8 text file, returning numbered lines. Large files are truncated; page with start_line.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the workspace root"},
                    "start_line": {"type": "integer", "description": "1-based first line to return. Defaults to 1"},
                    "max_lines": {"type": "integer", "description": "How many lines to return. Defaults to 200"},
                },
                "required": ["path"],
            },
            fn=workspace.read_file,
        ))
        .add(Tool(
            name="grep",
            description="Search file contents with a Python regular expression, returning file:line: text matches.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Python re pattern, for example 'TODO|FIXME'"},
                    "path": {"type": "string", "description": "File or directory to search. Defaults to '.'"},
                    "glob": {"type": "string", "description": "Filename filter when path is a directory. Defaults to '*'"}, 
                },
                "required": ["pattern"],
            },
            fn=workspace.grep,
        ))
    )
