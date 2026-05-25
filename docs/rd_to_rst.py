"""Convert man/*.Rd files to docs/python/r_api/*.rst for Sphinx."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
MAN_DIR = ROOT / "man"
OUT_DIR = Path(__file__).parent / "python" / "r_api"


def _strip_rd(text: str) -> str:
    """Remove simple Rd markup: \\code{x} → ``x``, \\code{NA} → ``NA``, etc."""
    text = re.sub(r"\\code\{([^}]*)\}", r"``\1``", text)
    text = re.sub(r"\\pkg\{([^}]*)\}", r"**\1**", text)
    text = re.sub(r"\\emph\{([^}]*)\}", r"*\1*", text)
    text = re.sub(r"\\strong\{([^}]*)\}", r"**\1**", text)
    text = re.sub(r"\\link\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", text)
    return text.strip()


def _extract(content: str, tag: str) -> str:
    """Extract the body of a \\tag{...} block (handles nested braces)."""
    pattern = f"\\{tag}{{"
    start = content.find(pattern)
    if start == -1:
        return ""
    idx = start + len(pattern)
    depth = 1
    chars: list[str] = []
    while idx < len(content) and depth > 0:
        ch = content[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        if depth > 0:
            chars.append(ch)
        idx += 1
    return "".join(chars).strip()


def _extract_items(content: str, tag: str) -> list[tuple[str, str]]:
    """Extract all \\item{name}{desc} pairs inside a \\tag{} block."""
    block = _extract(content, tag)
    items: list[tuple[str, str]] = []
    pos = 0
    while True:
        m = re.search(r"\\item\s*\{", block[pos:])
        if not m:
            break
        abs_start = pos + m.start() + len(m.group())
        # extract first brace group (name)
        depth = 1
        name_chars: list[str] = []
        i = abs_start
        while i < len(block) and depth > 0:
            ch = block[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            if depth > 0:
                name_chars.append(ch)
            i += 1
        name = "".join(name_chars).strip()
        i += 1  # skip closing }
        # extract second brace group (description)
        while i < len(block) and block[i] in (" ", "\n", "\r", "\t"):
            i += 1
        desc_chars: list[str] = []
        if i < len(block) and block[i] == "{":
            i += 1
            depth = 1
            while i < len(block) and depth > 0:
                ch = block[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        break
                if depth > 0:
                    desc_chars.append(ch)
                i += 1
        desc = "".join(desc_chars).strip()
        items.append((name, desc))
        pos = i + 1
    return items


def _indent(text: str, prefix: str) -> str:
    """Indent every line of text with prefix."""
    lines = text.splitlines()
    return "\n".join(prefix + l if l.strip() else "" for l in lines)


def rd_to_rst(rd_path: Path) -> str:
    content = rd_path.read_text(encoding="utf-8")

    title = _strip_rd(_extract(content, "title"))
    description = _strip_rd(_extract(content, "description"))
    usage = " ".join(_extract(content, "usage").split())  # collapse to one line
    value = " ".join(_strip_rd(_extract(content, "value")).split())  # collapse to one line
    examples_raw = _extract(content, "examples").strip()
    args = _extract_items(content, "arguments")

    rst = ""

    if usage:
        rst += f".. function:: {usage}\n\n"

    rst += _indent(f"*{title}* — {description}" if title else description, "   ") + "\n\n"

    for name, desc in args:
        cleaned = " ".join(_strip_rd(desc).split())  # collapse to one line
        rst += f"   :param {name}: {cleaned}\n"

    if value:
        rst += f"   :returns: {value}\n"

    if examples_raw:
        lines = [l for l in examples_raw.splitlines() if not l.strip().startswith("%")]
        example_code = "\n".join(lines).strip()
        if example_code:
            rst += "\n"
            rst += "   .. rubric:: Example\n\n"
            rst += "   .. code-block:: r\n\n"
            for line in example_code.splitlines():
                rst += f"      {line}\n"
            rst += "\n"

    return rst


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rd_files = sorted(MAN_DIR.glob("*.Rd"))
    rd_files = [f for f in rd_files if not f.stem.endswith("-package")]

    # Remove stale per-function .rst files from a previous run
    for old in OUT_DIR.glob("*.rst"):
        if old.name != "index.rst":
            old.unlink()

    index = "R API Reference\n===============\n\n"
    index += "All functions accept a numeric vector ``x`` (and ``y`` for bivariate\n"
    index += "functions), a ``window_size`` integer, and an optional ``min_periods``\n"
    index += "parameter compatible with *pandas* semantics.\n\n"

    for i, rd in enumerate(rd_files):
        index += rd_to_rst(rd)
        if i < len(rd_files) - 1:
            index += "\n----\n\n"
        print(f"  {rd.name} → r_api/index.rst")

    (OUT_DIR / "index.rst").write_text(index, encoding="utf-8")
    print(f"  -> r_api/index.rst ({len(rd_files)} functions)")


if __name__ == "__main__":
    sys.exit(main())
