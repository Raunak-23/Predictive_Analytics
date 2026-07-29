#!/usr/bin/env python3
"""
Surgically refine the two Lab-02 notebooks in place (idempotent).

* Markdown cells  -> standardised headings + de-mojibaked text.
* Code cells      -> a single concise purpose-comment header is prepended
                     (existing banner comments are stripped); every
                     executable statement is preserved verbatim.

Shared helpers imported by refine_breast.py / refine_diabetes.py.
"""

from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NB_DIR = os.path.join(ROOT, "notebooks")
SRC = os.path.join(ROOT, "src")
for p in (ROOT, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Mojibake / smart-punctuation cleanup (defined via unicode escapes so the
# helper file is pure ASCII and never trips the cp1252-on-Windows hazard).
# ---------------------------------------------------------------------------
MOJIBAKE = {
    "\u00e2\u20ac\u201c": "-",   # a-currency-dash -> dash  (was 'bad' en dash)
    "\u00e2\u20ac\u201d": "-",
    "\u00e2\u20ac\u2018": "'",
    "\u00e2\u20ac\u2019": "'",
    "\u00e2\u20ac\u201c": "-",
    "\u00e2\u20ac\u009d": '"',
    "\u00c2": "",                 # stray A-circumflex leftover
    "\u00c3\u00a9": "e",          # mojibaked accent-e
    "\u00c3\u00a8": "e",
    "\u00c3\u00a1": "a",
    "\u00c3\u00ad": "i",
    "\u00c3\u00b3": "o",
    "\u00c3\u00ba": "u",
    "\u00c3\u00b1": "n",
    "\u00c3\u00bc": "u",
    "\u00c3\u20ac": "e",
    "\u00c3\u00a7": "c",
    "\u00c3\u2030": "E",
    "\u00ce\u00b1": "alpha",      # greek alpha
    "\u00ce\u00b2": "beta",
    "\u00ce\u00b3": "gamma",
    "\u00ce\u00b4": "delta",
    "\u03bb": "lambda",
    "\u2212": "-",                # proper minus
    "\u2248": "~=",
    "\u2265": ">=",
    "\u2264": "<=",
    "\u2192": "->",
    # Direct en/em dashes and curly quotes that occasionally survive cp1252
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00b7": "-",
    # raw cp1252-decoded sequences the notebooks actually contained
    "\u00e2\u0080\u0093": "-",
    "\u00e2\u0080\u0094": "-",
    "\u00e2\u0080\u0098": "'",
    "\u00e2\u0080\u0099": "'",
    "\u00e2\u0080\u009c": '"',
    "\u00e2\u0080\u009d": '"',
    "\u00e2\u0080\u00a0": "-",
}
# Build the alternation longest-first so the multi-byte sequences are matched
# before any single-char fallbacks.
_keys = sorted(MOJIBAKE.keys(), key=len, reverse=True)
MOJI_RE = re.compile("|".join(re.escape(k) for k in _keys))


def demoji(s: str) -> str:
    return MOJI_RE.sub(lambda m: MOJIBAKE[m.group(0)], s)


# ---------------------------------------------------------------------------
# Notebook IO
# ---------------------------------------------------------------------------
def load_nb(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_nb(path, nb):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")


def src_str(cell) -> str:
    return "".join(cell.get("source", []))


def set_src(cell, text: str):
    text = text.rstrip("\n")
    if text == "":
        cell["source"] = []
    else:
        parts = text.split("\n")
        cell["source"] = [p + "\n" for p in parts[:-1]] + [parts[-1]]
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


# ---------------------------------------------------------------------------
# Banner stripping helpers (keep code executable statements intact)
# ---------------------------------------------------------------------------
_BANNER_RE = re.compile(r"^\s*#\s*={3,}\s*\n"
                        r"(?:\s*#.*\n)*?"
                        r"\s*#\s*={3,}\s*\n", re.MULTILINE)
_DASH_RE = re.compile(r"^\s*#\s*-{3,}\s*\n"
                      r"(?:\s*#.*\n)*?"
                      r"\s*#\s*-{3,}\s*\n", re.MULTILINE)


def strip_banner(src: str) -> str:
    m = _BANNER_RE.match(src)
    if m:
        return src[m.end():].lstrip("\n")
    m = _DASH_RE.match(src)
    if m:
        return src[m.end():].lstrip("\n")
    return src


def first_executable_line(src: str):
    for line in src.split("\n"):
        s = line.strip()
        if s and not s.startswith("#"):
            return line
    return None
