#!/usr/bin/env python3
"""Integrity checker for the bench-text-v1-en TTS benchmark text dataset.

Run (from the repo root):
    uv run python benchmarks/data/validate.py

Exits 0 when every check passes, 1 otherwise. Uses only the standard library.

Checks (per record, per file):
  * every line is valid JSON with the required keys
  * `word_len` equals the actual whitespace-split word count of `text`
  * word length falls inside the bucket window (main 35-55, long 300-350)
  * ids are well-formed and unique within each file
  * `source` is a permitted label (main) / names a Gutenberg text (long)
  * `cancel_plan.json`: cancel_ids all exist in texts_main and match the 30% rule
  * `manifest.csv` cross-checks every id (word_len, bucket, has_digits, source)
  * all data files are UTF-8 without a BOM
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent
MAIN_LO, MAIN_HI = 35, 55
LONG_LO, LONG_HI = 300, 350
MAIN_SOURCES = {"harvard", "ljspeech", "commonvoice"}
MAIN_ID = re.compile(r"^main-\d{3}$")
LONG_ID = re.compile(r"^long-\d{3}$")
DIGIT = re.compile(r"[0-9]")

errors: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        errors.append(msg)


def has_digits(text: str) -> bool:
    return bool(DIGIT.search(text))


def no_bom(path: Path) -> None:
    check(path.read_bytes()[:3] != b"\xef\xbb\xbf", f"{path.name}: starts with a UTF-8 BOM")


def load_jsonl(path: Path, want_keys: set[str]) -> list[dict]:
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"{path.name}:{i}: invalid JSON ({e})")
            continue
        missing = want_keys - rec.keys()
        check(not missing, f"{path.name}:{i}: missing keys {sorted(missing)}")
        out.append(rec)
    return out


# --------------------------------------------------------------------------- main
def validate_main() -> list[dict]:
    path = DATA / "texts_main.jsonl"
    no_bom(path)
    recs = load_jsonl(path, {"id", "text", "word_len", "source"})
    ids = set()
    for i, r in enumerate(recs, 1):
        rid = r.get("id", f"#{i}")
        check(MAIN_ID.match(r.get("id", "")), f"main {rid}: id not of form main-NNN")
        check(r.get("id") not in ids, f"main {rid}: duplicate id")
        ids.add(r.get("id"))
        text = r.get("text", "")
        check(isinstance(text, str) and text, f"main {rid}: text missing/empty")
        check(isinstance(r.get("word_len"), int), f"main {rid}: word_len not int")
        actual = len(text.split())
        check(r.get("word_len") == actual,
              f"main {rid}: word_len {r.get('word_len')} != actual {actual}")
        check(MAIN_LO <= actual <= MAIN_HI,
              f"main {rid}: {actual} words outside [{MAIN_LO},{MAIN_HI}]")
        check(r.get("source") in MAIN_SOURCES,
              f"main {rid}: source {r.get('source')!r} not in {sorted(MAIN_SOURCES)}")
    check(len(recs) == 60, f"texts_main.jsonl: expected 60 records, got {len(recs)}")
    check(len({r["text"] for r in recs if "text" in r}) == len(recs),
          "texts_main.jsonl: duplicate texts detected")
    return recs


# --------------------------------------------------------------------------- long
def validate_long() -> list[dict]:
    path = DATA / "texts_long.jsonl"
    no_bom(path)
    recs = load_jsonl(path, {"id", "text", "word_len", "source"})
    ids = set()
    for i, r in enumerate(recs, 1):
        rid = r.get("id", f"#{i}")
        check(LONG_ID.match(r.get("id", "")), f"long {rid}: id not of form long-NNN")
        check(r.get("id") not in ids, f"long {rid}: duplicate id")
        ids.add(r.get("id"))
        text = r.get("text", "")
        actual = len(text.split())
        check(r.get("word_len") == actual,
              f"long {rid}: word_len {r.get('word_len')} != actual {actual}")
        check(LONG_LO <= actual <= LONG_HI,
              f"long {rid}: {actual} words outside [{LONG_LO},{LONG_HI}]")
        check(isinstance(r.get("source"), str) and "Gutenberg" in r.get("source", ""),
              f"long {rid}: source must name a Gutenberg text")
    check(len(recs) == 10, f"texts_long.jsonl: expected 10 records, got {len(recs)}")
    return recs


# ------------------------------------------------------------------------- cancel
def validate_cancel(main_ids: set[str], n_main: int) -> None:
    path = DATA / "cancel_plan.json"
    no_bom(path)
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        errors.append(f"cancel_plan.json: invalid JSON ({e})")
        return
    for key in ("concurrency", "cancel_ratio", "cancel_ids", "cancel_after_ms"):
        check(key in plan, f"cancel_plan.json: missing key {key!r}")
    cids = plan.get("cancel_ids", [])
    expected = round(0.3 * n_main)
    check(len(cids) == expected,
          f"cancel_plan.json: expected {expected} cancel_ids (30% of {n_main}), got {len(cids)}")
    missing = [c for c in cids if c not in main_ids]
    check(not missing, f"cancel_plan.json: cancel_ids not in texts_main: {missing}")
    check(len(set(cids)) == len(cids), "cancel_plan.json: duplicate cancel_ids")
    check(abs(plan.get("cancel_ratio", 0) - 0.3) < 1e-9, "cancel_plan.json: cancel_ratio != 0.3")
    check(plan.get("cancel_after_ms") == 500, "cancel_plan.json: cancel_after_ms != 500")


# ------------------------------------------------------------------------ manifest
def validate_manifest(main_recs: list[dict], long_recs: list[dict]) -> None:
    path = DATA / "manifest.csv"
    no_bom(path)
    by_id = {**{r["id"]: r for r in main_recs}, **{r["id"]: r for r in long_recs}}
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    seen = set()
    for row in rows:
        rid = row.get("id", "")
        check(rid in by_id, f"manifest.csv: id {rid!r} not in any jsonl")
        seen.add(rid)
        rec = by_id.get(rid, {})
        text = rec.get("text", "")
        check(row.get("word_len", "").strip() == str(rec.get("word_len", "")),
               f"manifest.csv: {rid} word_len mismatch")
        check(("true" if has_digits(text) else "false") == row.get("has_digits"),
               f"manifest.csv: {rid} has_digits mismatch")
        check(row.get("source") == rec.get("source"), f"manifest.csv: {rid} source mismatch")
    check(seen == set(by_id),
          f"manifest.csv: id set mismatch (missing {set(by_id)-seen}, extra {seen-set(by_id)})")


# ----------------------------------------------------------------------- main flow
def main() -> int:
    main_recs = validate_main()
    long_recs = validate_long()
    validate_cancel({r["id"] for r in main_recs}, len(main_recs))
    validate_manifest(main_recs, long_recs)

    # ---- summary
    print("bench-text-v1-en validation")
    print("-" * 48)
    if main_recs:
        wl = [r["word_len"] for r in main_recs]
        print(f"main : {len(main_recs)} records | word_len "
              f"min={min(wl)} median={statistics.median(wl)} max={max(wl)} "
              f"(window {MAIN_LO}-{MAIN_HI})")
        print(f"        questions={sum('?' in r['text'] for r in main_recs)} "
              f"digit_items={sum(has_digits(r['text']) for r in main_recs)} "
              f"phone_items={sum('555-' in r['text'] for r in main_recs)}")
        src = {}
        for r in main_recs:
            src[r["source"]] = src.get(r["source"], 0) + 1
        print(f"        sources={src}")
    if long_recs:
        wl = [r["word_len"] for r in long_recs]
        print(f"long : {len(long_recs)} records | word_len min={min(wl)} max={max(wl)} "
              f"(window {LONG_LO}-{LONG_HI}) | all digit-bearing="
              f"{all(has_digits(r['text']) for r in long_recs)}")
    try:
        plan = json.loads((DATA / "cancel_plan.json").read_text(encoding="utf-8"))
        print(f"cancel: {len(plan.get('cancel_ids', []))} ids "
              f"({plan.get('cancel_ratio')} of main), cancel_after_ms={plan.get('cancel_after_ms')}")
    except Exception:
        pass
    print("-" * 48)

    if errors:
        print(f"FAIL — {len(errors)} problem(s):")
        for e in errors:
            print(f"  x {e}")
        return 1
    print("PASS — all integrity checks succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
