"""
Prepare normalized CSV files for D2 (Enron-Spam) and D3 (SpamAssassin).

This is a one-shot data-preparation script. Run it once to materialize the
two public spam datasets into the unified schema required by the lab manual:
    email_id, subject, body, label, dataset_id

Outputs:
    ../data/d2_d3_normalized/enron_spam.csv
    ../data/d2_d3_normalized/spamassassin.csv

Sources:
    D2 - HuggingFace: SetFit/enron_spam  (Enron-Spam corpus, ~33k rows)
    D3 - HuggingFace: bvk/SpamAssassin-spam (SpamAssassin public corpus)

Both datasets are publicly available and licensed for research / education.
The original CSV files already in this project (`spam_email_dataset_*.csv`)
were sampled from a similar Enron-derived corpus; we still create the
normalized file from the canonical Enron-Spam source so that the workflow
is reproducible across machines.
"""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "d2_d3_normalized"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

_SUBJECT_RE = re.compile(r"^\s*Subject:\s*(.+?)\s*(?:\\n|\n|$)", re.IGNORECASE)
# Pattern used to strip the Python-literal list wrapper that some HF
# datasets use to store raw message arrays, e.g. ["Subject: ...\\n..."]
_LIST_LITERAL_RE = re.compile(r'^\s*\[\s*(.*?)\s*\]\s*$', re.DOTALL)
_ESCAPED_NEWLINE_RE = re.compile(r"\\n")


def _normalize_raw(raw: object) -> str:
    """Convert whatever the HF dataset returns into a single string and
    unescape literal ``\\n`` sequences that some datasets use to encode
    newlines inside a stringified list."""
    if isinstance(raw, list):
        text = "\n".join(str(x) for x in raw)
    else:
        text = str(raw or "")
    text = _ESCAPED_NEWLINE_RE.sub("\n", text)
    return text


def extract_subject_and_body(raw: object) -> tuple[str, str]:
    """Split a raw email into (subject, body). The Enron-Spam and SpamAssassin
    corpora often begin with a `Subject:` header followed by the message body.
    The function tolerates Python-list-wrapped strings and falls back to a
    naive split when no header is detected.
    """
    text = _normalize_raw(raw).strip()
    if not text:
        return "", ""
    # Strip the wrapping list literal if present, e.g. ["Subject: foo\nbar"]
    list_match = _LIST_LITERAL_RE.match(text)
    if list_match:
        text = list_match.group(1)
        text = "\n".join(line.strip().strip("'\"") for line in text.splitlines())
    match = _SUBJECT_RE.search(text)
    if match:
        subject = match.group(1).strip()
        body = text[match.end():].strip()
    else:
        subject = ""
        body = text
    return subject, body


def make_email_id(dataset_id: str, idx: int, subject: str, body: str) -> str:
    """Deterministic anonymized ID for each email."""
    digest = hashlib.sha256(f"{dataset_id}|{idx}|{subject}|{body}".encode("utf-8")).hexdigest()
    return f"{dataset_id}_{digest[:16]}"


# ---------------------------------------------------------------------------
# Enron-Spam (D2)
# ---------------------------------------------------------------------------

def prepare_d2_enron_spam() -> Path:
    print("[D2] Loading SetFit/enron_spam ...")
    ds = load_dataset("SetFit/enron_spam", split="train")
    rows = []
    for i, ex in enumerate(ds):
        # Enron-Spam fields: 'subject' (or empty) and 'message' (or 'text')
        subject = (ex.get("subject") or "").strip()
        body = (ex.get("message") or ex.get("text") or "").strip()
        if not body and not subject:
            continue
        label_text = (ex.get("label_text") or "").strip().lower()
        label = "spam" if label_text == "spam" else "legitimate"
        rows.append({
            "subject": subject,
            "body": body,
            "label": label,
        })
    df = pd.DataFrame(rows)
    df["email_id"] = [make_email_id("enron_spam", i, r.subject, r.body)
                      for i, r in df.iterrows()]
    df["dataset_id"] = "enron_spam"
    df = df[["email_id", "subject", "body", "label", "dataset_id"]]
    out_path = DATA_DIR / "enron_spam.csv"
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[D2] Saved {len(df)} rows -> {out_path}")
    print(df["label"].value_counts())
    return out_path


# ---------------------------------------------------------------------------
# SpamAssassin (D3)
# ---------------------------------------------------------------------------

def prepare_d3_spamassassin() -> Path:
    print("[D3] Loading bvk/SpamAssassin-spam ...")
    ds = load_dataset("bvk/SpamAssassin-spam", split="train")
    rows = []
    for i, ex in enumerate(ds):
        raw_text = ex.get("data")
        subject, body = extract_subject_and_body(raw_text)
        if not body and not subject:
            continue
        label = "spam" if int(ex.get("label", 0)) == 1 else "legitimate"
        rows.append({
            "subject": subject,
            "body": body,
            "label": label,
        })
    df = pd.DataFrame(rows)
    df["email_id"] = [make_email_id("spamassassin", i, r.subject, r.body)
                      for i, r in df.iterrows()]
    df["dataset_id"] = "spamassassin"
    df = df[["email_id", "subject", "body", "label", "dataset_id"]]
    out_path = DATA_DIR / "spamassassin.csv"
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[D3] Saved {len(df)} rows -> {out_path}")
    print(df["label"].value_counts())
    return out_path


if __name__ == "__main__":
    prepare_d2_enron_spam()
    prepare_d3_spamassassin()
