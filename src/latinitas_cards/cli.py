import csv
import hashlib
import html
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Annotated
from urllib import error as urlerror
from urllib import request as urlrequest

import pandas as pd
import typer
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

stderr_console = Console(stderr=True)
stdout_console = Console()


def info(message: str) -> None:
    stderr_console.print(f"[bold cyan][INFO][/bold cyan] {message}")


def success(message: str) -> None:
    stderr_console.print(f"[bold green][OK][/bold green] {message}")


def _status_label(ok: bool) -> str:
    return "[green]PASS[/green]" if ok else "[red]FAIL[/red]"


def _render_validation_table(checks: list[dict[str, str]]) -> None:
    table = Table(title="Validation Report", box=box.SIMPLE_HEAVY)
    table.add_column("Check", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Details", overflow="fold")
    for check in checks:
        table.add_row(check["name"], check["status"], check["details"])
    stdout_console.print(table)


_BLOCK_TAG_RE = re.compile(r"<(div|br|p|li|ul|ol|tr|td|th|table|blockquote|h[1-6])\b", flags=re.IGNORECASE)
_BLOCK_BREAK_TAG_RE = re.compile(
    r"</?(div|br|p|li|ul|ol|tr|td|th|table|blockquote|h[1-6])\b[^>]*>", flags=re.IGNORECASE
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_ENTITY_MAP: dict[str, str] = {
    "&nbsp;": " ",
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
}


def strip_anki_field(s: str) -> str:
    """Strip HTML tags, HTML entities, and Unicode macrons/breves from an Anki field value.

    Block-level HTML elements (``<div>``, ``<br>``, etc.) and everything after
    them are dropped — in typical Anki exports the main word/phrase appears
    before such elements, while hints and extra info follow inside them.
    """
    return clean_anki_field(s, truncate_at_block=True)


def clean_anki_field(s: str, truncate_at_block: bool) -> str:
    """Normalize Anki HTML-ish content to plain text.

    When ``truncate_at_block`` is true, content after the first block-level tag is
    dropped (the historical behavior for matching front-side terms).
    """
    if not s:
        return s
    if truncate_at_block:
        m = _BLOCK_TAG_RE.search(s)
        if m:
            s = s[: m.start()]
    else:
        s = _BLOCK_BREAK_TAG_RE.sub(" | ", s)
    for entity, char in _HTML_ENTITY_MAP.items():
        s = s.replace(entity, char)
    s = html.unescape(s)
    s = _HTML_TAG_RE.sub("", s)
    s = re.sub(r"\s*\|\s*", " | ", s)
    s = re.sub(r"(?:\s*\|\s*){2,}", " | ", s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" |")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return s.strip()


def normalize_latin(s: str) -> str:
    if s is None:
        return ""
    s = s.lower().strip()
    s = s.replace("æ", "ae").replace("œ", "oe")
    s = s.replace("j", "i")
    return s


def make_word_regex(word: str) -> re.Pattern[str]:
    w = re.escape(normalize_latin(word))
    return re.compile(rf"\b{w}\b", flags=re.IGNORECASE)


def cloze_once(text: str, pattern: re.Pattern[str]) -> tuple[str, int]:
    def repl(m: re.Match[str]) -> str:
        return "{{c1::" + m.group(0) + "}}"

    new_text, n = pattern.subn(repl, text, count=1)
    return new_text, n


def cloze_all(text: str, pattern: re.Pattern[str]) -> tuple[str, int]:
    def repl(m: re.Match[str]) -> str:
        return "{{c1::" + m.group(0) + "}}"

    return pattern.subn(repl, text)


def parse_usfx_to_df(path: Path) -> pd.DataFrame:
    tree = ET.parse(path)
    root = tree.getroot()

    rows: list[dict[str, object]] = []

    current_book: str | None = None
    current_chapter: int | None = None
    current_verse: int | None = None
    buffer: list[str] = []

    number_re = re.compile(r"\d+")

    def norm_space(s: str) -> str:
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def flush() -> None:
        nonlocal current_verse, buffer
        if current_book and current_chapter is not None and current_verse is not None:
            txt = norm_space(" ".join(part for part in buffer if part and part.strip()))
            if txt:
                rows.append(
                    {
                        "book": current_book,
                        "chapter": int(current_chapter),
                        "verse": int(current_verse),
                        "text": txt,
                    }
                )
        buffer = []
        current_verse = None

    # We need to walk in document order and grab text residing outside tags too.
    # xml.etree doesn't give easy "between siblings" text via iter(), so we do a manual stack walk
    # capturing both .text and .tail relative to verse boundaries.
    stack: list[tuple[ET.Element, int]] = [(root, 0)]
    while stack:
        node, state = stack.pop()

        # On first visit, process start of node, then push children, then mark to process tail after children
        if state == 0:
            tag = node.tag.lower()

            if tag in ("book", "b"):
                flush()
                current_book = node.attrib.get("id") or node.attrib.get("code") or node.attrib.get("n") or current_book

            elif tag in ("c", "chapter"):
                flush()
                ch_val = node.attrib.get("id") or node.attrib.get("n") or ""
                m = number_re.search(ch_val)
                current_chapter = int(m.group()) if m else current_chapter

            elif tag in ("v", "verse"):
                # Start a new verse; if one is open, flush it first
                flush()
                v_val = node.attrib.get("id") or node.attrib.get("n") or ""
                m = number_re.search(v_val)
                current_verse = int(m.group()) if m else 0
                # Initial text inside <v>...</v> if any
                if node.text and current_verse is not None:
                    buffer.append(node.text)

            elif tag == "ve":
                # Verse end; include any text in <ve> (rare) then flush
                if node.text and current_verse is not None:
                    buffer.append(node.text)
                flush()

            else:
                # Normal node text inside an open verse
                if node.text and current_verse is not None:
                    buffer.append(node.text)

            # push a marker to process tail after children
            stack.append((node, 1))
            # push children in reverse to simulate pre-order traversal
            children = list(node)
            for child in reversed(children):
                stack.append((child, 0))

        else:
            # state == 1: process tail text after closing this node
            if node.tail and current_verse is not None:
                buffer.append(node.tail)

    # End of document: flush any open verse
    flush()

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Parsed zero verses from USFX; check that <v/> ... text ... <ve/> structure is present.")

    df["text_norm"] = df["text"].apply(normalize_latin)
    return df.sort_values(["book", "chapter", "verse"]).reset_index(drop=True)  # type: ignore[no-any-return]


def build_bucket_index(df: pd.DataFrame) -> dict[str, list[int]]:
    bucket: dict[str, list[int]] = defaultdict(list)
    for pos, (_, row) in enumerate(df.iterrows()):
        tn = str(row["text_norm"])
        if not tn:
            continue
        letters = set(re.findall(r"[a-z]", tn[:60])) or {"*"}
        for ch in letters:
            bucket[ch].append(pos)
    return bucket


def candidate_indices(word_norm: str, bucket: dict[str, list[int]], total_len: int) -> list[int] | range:
    first = next((c for c in word_norm if c.isalpha()), "*")
    return bucket.get(first, range(total_len))


def read_stopwords(path: Path) -> set[str]:
    stops: set[str] = set()
    if not path:
        return stops
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            stops.add(normalize_latin(t))
    return stops


def read_lemma_index(path: Path) -> dict[str, set[str]]:
    """Read lemma/form groups and build lookup index keyed by normalized form.

    Accepted line formats are flexible, for example:
    - ``amo, amas, amat``
    - ``amo: amas amat``
    - ``amo => amas amat``
    """
    lemma_index: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("#"):
                continue
            tokens = [normalize_latin(t) for t in re.split(r"(?:=>|[:,]|\s)+", cleaned) if t.strip()]
            forms = {t for t in tokens if t}
            if not forms:
                continue
            for form in forms:
                lemma_index[form] = forms
    return lemma_index


def compile_ignore_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, flags=re.IGNORECASE) for p in patterns if p.strip()]


def generate_clozes_for_word(
    df: pd.DataFrame,
    word: str,
    bucket: dict[str, list[int]],
    max_examples: int = 2,
    multi_cloze_per_verse: bool = False,
    lookup_terms: set[str] | None = None,
) -> list[str]:
    word_norm = normalize_latin(word)
    terms = lookup_terms or {word_norm}
    out = []
    cnt = 0
    seen: set[int] = set()
    for term in terms:
        patt = make_word_regex(term)
        for idx in candidate_indices(term, bucket, len(df)):
            if idx in seen:
                continue
            verse_text = str(df.at[idx, "text"])
            verse_norm = str(df.at[idx, "text_norm"])
            if term not in verse_norm:
                continue
            if multi_cloze_per_verse:
                cloze, n = cloze_all(verse_text, patt)
            else:
                cloze, n = cloze_once(verse_text, patt)
            if n > 0:
                if {"book", "chapter", "verse"}.issubset(df.columns):
                    ref = f"{df.at[idx, 'book']} {df.at[idx, 'chapter']}:{df.at[idx, 'verse']}"
                elif "ref" in df.columns:
                    ref = str(df.at[idx, "ref"])
                else:
                    ref = str(df.at[idx, "source_path"]) if "source_path" in df.columns else str(idx)
                out.append(f"{cloze} <span style='color:#888'>({ref})</span>")
                cnt += 1
                seen.add(int(idx))
                if cnt >= max_examples:
                    return out
    return out


def collect_cloze_matches(
    df: pd.DataFrame,
    word: str,
    bucket: dict[str, list[int]],
    max_examples: int,
    multi_cloze_per_verse: bool,
    translation_columns: list[str],
) -> list[dict[str, str]]:
    """Collect cloze matches together with refs and optional translations."""
    term = normalize_latin(word)
    if not term:
        return []
    patt = make_word_regex(term)
    matches: list[dict[str, str]] = []
    seen: set[int] = set()
    for idx in candidate_indices(term, bucket, len(df)):
        if idx in seen:
            continue
        verse_text = str(df.at[idx, "text"])
        verse_norm = str(df.at[idx, "text_norm"])
        if term not in verse_norm:
            continue
        if multi_cloze_per_verse:
            cloze, n = cloze_all(verse_text, patt)
        else:
            cloze, n = cloze_once(verse_text, patt)
        if n <= 0:
            continue
        if {"book", "chapter", "verse"}.issubset(df.columns):
            ref = f"{df.at[idx, 'book']} {df.at[idx, 'chapter']}:{df.at[idx, 'verse']}"
        elif "ref" in df.columns:
            ref = str(df.at[idx, "ref"])
        else:
            ref = str(idx)
        row: dict[str, str] = {
            "cloze": cloze,
            "ref": ref,
            "sentence_norm": verse_norm,
        }
        for column in translation_columns:
            if column in df.columns:
                row[column] = str(df.at[idx, column])
        matches.append(row)
        seen.add(int(idx))
        if len(matches) >= max_examples:
            break
    return matches


def _pick_collection_database_name(names: list[str]) -> tuple[str, bool]:
    sqlite_name = next((name for name in names if name.endswith(".anki21b")), None)
    if sqlite_name:
        return sqlite_name, True
    sqlite_name = next((name for name in names if name.endswith(".anki2") or name.endswith(".anki21")), None)
    if sqlite_name:
        return sqlite_name, False
    raise ValueError("APKG does not contain a collection database (.anki2/.anki21/.anki21b).")


def _safe_row_int(row: sqlite3.Row, column: str, default: int = 0) -> int:
    try:
        return int(row[column])
    except Exception:
        return default


def _load_field_names_by_mid(con: sqlite3.Connection) -> dict[int, list[str]]:
    mapping: dict[int, list[str]] = {}
    try:
        rows = con.execute("SELECT ntid, ord, name FROM fields ORDER BY ntid, ord").fetchall()
        if rows:
            for ntid, _, name in rows:
                mapping.setdefault(int(ntid), []).append(str(name))
            return mapping
    except sqlite3.Error:
        pass

    try:
        meta = con.execute("SELECT models FROM col LIMIT 1").fetchone()
        if not meta or not meta[0]:
            return mapping
        models = json.loads(meta[0])
        if not isinstance(models, dict):
            return mapping
        for mid, model in models.items():
            if not isinstance(model, dict) or "flds" not in model:
                continue
            fields: list[str] = []
            for item in model["flds"]:
                if isinstance(item, dict):
                    fields.append(str(item.get("name", "")))
            mapping[int(mid)] = fields
    except (sqlite3.Error, ValueError, TypeError, KeyError):
        return mapping

    return mapping


def _load_notetype_names(con: sqlite3.Connection) -> dict[int, str]:
    names: dict[int, str] = {}
    try:
        rows = con.execute("SELECT id, name FROM notetypes").fetchall()
        for ntid, name in rows:
            names[int(ntid)] = str(name)
        if names:
            return names
    except sqlite3.Error:
        pass

    try:
        meta = con.execute("SELECT models FROM col LIMIT 1").fetchone()
        if not meta or not meta[0]:
            return names
        models = json.loads(meta[0])
        if not isinstance(models, dict):
            return names
        for mid, model in models.items():
            if isinstance(model, dict):
                names[int(mid)] = str(model.get("name", mid))
    except (sqlite3.Error, ValueError, TypeError, KeyError):
        return names
    return names


def _resolve_note_field_index(
    con: sqlite3.Connection,
    field_name: str,
    fallback_to_front: bool,
    mid: int | None = None,
    field_names_by_mid: dict[int, list[str]] | None = None,
) -> int:
    candidate_index = 0
    front_like_names = ["front", "expression", "word", "latein"]
    names_by_mid = field_names_by_mid or _load_field_names_by_mid(con)

    selected_names: list[str] | None = None
    if mid is not None and mid in names_by_mid:
        selected_names = names_by_mid[mid]
    elif names_by_mid:
        selected_names = next(iter(names_by_mid.values()))

    if not selected_names:
        if fallback_to_front:
            return candidate_index
        raise KeyError(f"Field '{field_name}' not found in APKG note model.")

    lowered = [n.lower() for n in selected_names]
    target = field_name.lower()
    if target in lowered:
        return lowered.index(target)

    if fallback_to_front:
        for name in front_like_names:
            if name in lowered:
                return lowered.index(name)
        return candidate_index
    raise KeyError(f"Field '{field_name}' not found in APKG note model.")


def _read_apkg_field_rows(apkg_path: Path, field_name: str) -> list[dict[str, str]]:
    """Read notes from an .apkg/.colpkg and extract one field by name."""
    rows: list[dict[str, str]] = []

    with zipfile.ZipFile(apkg_path, "r") as zf, tempfile.TemporaryDirectory() as td:
        sqlite_name, is_zstd = _pick_collection_database_name(zf.namelist())
        db_path = Path(td) / "collection.anki2"
        raw = zf.read(sqlite_name)
        if is_zstd:
            import zstandard

            raw = zstandard.ZstdDecompressor().decompress(raw, max_output_size=256 * 1024 * 1024)
        with open(db_path, "wb") as dst:
            dst.write(raw)

        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
        try:
            names_by_mid = _load_field_names_by_mid(con)
            index_cache: dict[int, int] = {}
            cur = con.execute("SELECT mid, flds FROM notes")
            for r in cur:
                mid = _safe_row_int(r, "mid", default=0)
                if mid not in index_cache:
                    index_cache[mid] = _resolve_note_field_index(
                        con,
                        field_name,
                        fallback_to_front=True,
                        mid=mid,
                        field_names_by_mid=names_by_mid,
                    )
                candidate_index = index_cache[mid]
                parts = r["flds"].split("\x1f")
                if candidate_index < len(parts):
                    rows.append({field_name: parts[candidate_index]})
            return rows
        finally:
            con.close()


def _read_apkg_notes_dataframe(apkg_path: Path, notetype: str | None = None) -> pd.DataFrame:
    """Load APKG notes as a DataFrame with metadata and all named fields."""
    with zipfile.ZipFile(apkg_path, "r") as zf, tempfile.TemporaryDirectory() as td:
        sqlite_name, is_zstd = _pick_collection_database_name(zf.namelist())
        db_path = Path(td) / "collection.anki2"
        raw = zf.read(sqlite_name)
        if is_zstd:
            import zstandard

            raw = zstandard.ZstdDecompressor().decompress(raw, max_output_size=512 * 1024 * 1024)
        with open(db_path, "wb") as dst:
            dst.write(raw)

        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
        try:
            names_by_mid = _load_field_names_by_mid(con)
            notetype_names = _load_notetype_names(con)
            out: list[dict[str, str]] = []
            for row in con.execute("SELECT id, mid, flds FROM notes ORDER BY id"):
                note_mid = _safe_row_int(row, "mid", default=0)
                note_type_name = notetype_names.get(note_mid, str(note_mid))
                if notetype and note_type_name != notetype:
                    continue
                flds = str(row["flds"]).split("\x1f")
                names = names_by_mid.get(note_mid, [])
                built: dict[str, str] = {
                    "note_id": str(row["id"]),
                    "note_mid": str(note_mid),
                    "note_type": note_type_name,
                }
                for idx, value in enumerate(flds):
                    field_name = names[idx] if idx < len(names) and names[idx] else f"field_{idx}"
                    built[field_name] = str(value)
                out.append(built)
            return pd.DataFrame(out)
        finally:
            con.close()


def _update_apkg_with_cloze(
    apkg_input: Path,
    apkg_output: Path,
    usfx_path: Path,
    front_col: str,
    new_field: str,
    max_examples: int = 2,
    joiner: str = "<br><br>",
    stopwords_path: Path | None = None,
    overwrite: bool = True,
) -> None:
    with stderr_console.status("Loading corpus XML..."):
        bible_df = parse_usfx_to_df(usfx_path)
        bucket = build_bucket_index(bible_df)
    info(f"Loaded corpus: {usfx_path}")

    stopwords = read_stopwords(stopwords_path) if stopwords_path else set()
    if stopwords:
        info(f"Loaded {len(stopwords)} stopwords.")

    with zipfile.ZipFile(apkg_input, "r") as zf, tempfile.TemporaryDirectory() as td:
        sqlite_name, is_zstd = _pick_collection_database_name(zf.namelist())
        db_path = Path(td) / "collection.anki2"
        raw = zf.read(sqlite_name)
        if is_zstd:
            import zstandard

            raw = zstandard.ZstdDecompressor().decompress(raw, max_output_size=256 * 1024 * 1024)
        with open(db_path, "wb") as dst:
            dst.write(raw)

        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
        try:
            names_by_mid = _load_field_names_by_mid(con)
            notes = con.execute("SELECT id, mid, flds FROM notes").fetchall()
            index_cache: dict[tuple[int, str, bool], int] = {}

            missing_target_mid: set[int] = set()
            mids_with_notes = {_safe_row_int(note, "mid", default=0) for note in notes}
            for mid in mids_with_notes:
                try:
                    _resolve_note_field_index(
                        con,
                        new_field,
                        fallback_to_front=False,
                        mid=mid,
                        field_names_by_mid=names_by_mid,
                    )
                except KeyError:
                    missing_target_mid.add(mid)
            if missing_target_mid == mids_with_notes and mids_with_notes:
                raise KeyError(f"Field '{new_field}' not found in APKG note model.")

            updated = 0
            for note in notes:
                note_mid = _safe_row_int(note, "mid", default=0)
                if note_mid in missing_target_mid:
                    continue
                front_cache_key = (note_mid, front_col, True)
                if front_cache_key not in index_cache:
                    index_cache[front_cache_key] = _resolve_note_field_index(
                        con,
                        front_col,
                        fallback_to_front=True,
                        mid=note_mid,
                        field_names_by_mid=names_by_mid,
                    )
                target_cache_key = (note_mid, new_field, False)
                if target_cache_key not in index_cache:
                    index_cache[target_cache_key] = _resolve_note_field_index(
                        con,
                        new_field,
                        fallback_to_front=False,
                        mid=note_mid,
                        field_names_by_mid=names_by_mid,
                    )
                front_idx = index_cache[front_cache_key]
                target_idx = index_cache[target_cache_key]
                fields = note["flds"].split("\x1f")
                if front_idx >= len(fields) or target_idx >= len(fields):
                    continue
                front_val = strip_anki_field(fields[front_idx])
                if not front_val or normalize_latin(front_val) in stopwords:
                    new_val = ""
                else:
                    clozes = generate_clozes_for_word(bible_df, front_val, bucket, max_examples=max_examples)
                    new_val = joiner.join(clozes)

                if overwrite:
                    fields[target_idx] = new_val
                elif fields[target_idx].strip() and new_val.strip():
                    fields[target_idx] = fields[target_idx].strip() + joiner + new_val.strip()
                else:
                    fields[target_idx] = (fields[target_idx] or "") + (new_val or "")

                con.execute("UPDATE notes SET flds = ? WHERE id = ?", ("\x1f".join(fields), note["id"]))
                updated += 1

            con.commit()
        finally:
            con.close()

        updated_raw = db_path.read_bytes()
        if is_zstd:
            import zstandard

            updated_raw = zstandard.ZstdCompressor().compress(updated_raw)

        with zipfile.ZipFile(apkg_output, "w") as out_zip:
            for info_obj in zf.infolist():
                data = updated_raw if info_obj.filename == sqlite_name else zf.read(info_obj.filename)
                out_zip.writestr(info_obj, data)

    success(f"Updated {updated} notes and wrote: {apkg_output.resolve()}")


def _load_input_to_dataframe(input_path: Path, front_col: str) -> pd.DataFrame:
    """
    Load either CSV (with header) or APKG into a DataFrame exposing the front_col.
    """
    suffix = input_path.suffix.lower()
    if suffix in {".apkg", ".colpkg"}:
        apkg_rows = _read_apkg_field_rows(input_path, front_col)
        if not apkg_rows:
            raise ValueError("No notes found in APKG or field could not be resolved.")
        return pd.DataFrame(apkg_rows)
    else:
        # CSV path (existing behavior)
        with open(input_path, newline="", encoding="utf-8") as f:
            sample = f.read(4096)
            dialect = csv.Sniffer().sniff(sample)
            has_header = csv.Sniffer().has_header(sample)
        if not has_header:
            raise ValueError("CSV seems to have no header row. Please export with headers (Front, Back, ...).")
        separator = str(dialect.delimiter) if dialect.delimiter else ","
        return pd.read_csv(input_path, encoding="utf-8", sep=separator, keep_default_na=False)


def _load_any_input_dataframe(input_path: Path, notetype: str | None = None) -> pd.DataFrame:
    """Load CSV/APKG input with broad schema retention for inspect/split flows."""
    suffix = input_path.suffix.lower()
    if suffix in {".apkg", ".colpkg"}:
        return _read_apkg_notes_dataframe(input_path, notetype=notetype)
    return pd.read_csv(input_path, encoding="utf-8", keep_default_na=False)


def _list_notetype_fields(con: sqlite3.Connection) -> list[tuple[int, str, list[str]]]:
    names_by_mid = _load_field_names_by_mid(con)
    notetype_names = _load_notetype_names(con)
    out: list[tuple[int, str, list[str]]] = []
    for mid, fields in sorted(names_by_mid.items(), key=lambda item: item[0]):
        out.append((mid, notetype_names.get(mid, str(mid)), fields))
    return out


def _token_is_latin_like(token: str) -> bool:
    return bool(re.search(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ž]", token))


def _normalize_form_token(token: str) -> str:
    t = token.strip()
    if not t:
        return ""
    t = t.replace("(", "").replace(")", "")
    t = t.strip(" .,:;!?-–—")
    t = re.sub(r"\s+", " ", t)
    if not t:
        return ""
    parts = t.split(" ")
    first_latin = next((part for part in parts if _token_is_latin_like(part)), "")
    if not first_latin:
        return ""
    first_latin = first_latin.strip(" .,:;!?-–—")
    if re.fullmatch(r"[fmn]\.?", first_latin.lower()):
        return ""
    return first_latin


_SPLIT_REGEX: dict[str, str] = {
    "comma": r"\s*,\s*",
    "slash": r"\s*/\s*",
    "pipe": r"\s*\|\s*",
    "semicolon": r"\s*;\s*",
}


def split_latin_forms(
    value: str, mode: str = "auto", custom_separator_regex: str | None = None
) -> tuple[list[str], str, float]:
    """Split a card field into single lexical forms."""
    cleaned = clean_anki_field(value, truncate_at_block=False)
    if not cleaned:
        return [], "none", 0.0

    def split_with_pattern(pattern: str) -> list[str]:
        chunks = [chunk for chunk in re.split(pattern, cleaned) if chunk and chunk.strip()]
        forms = [_normalize_form_token(chunk) for chunk in chunks]
        unique = list(dict.fromkeys([f for f in forms if f]))
        return unique

    if mode == "custom":
        if not custom_separator_regex:
            raise ValueError("--custom-separator-regex is required when --split-mode=custom")
        items = split_with_pattern(custom_separator_regex)
        return (items or [_normalize_form_token(cleaned)]), "custom", 1.0 if len(items) > 1 else 0.6

    if mode != "auto":
        pattern = _SPLIT_REGEX.get(mode)
        if not pattern:
            raise ValueError(f"Unsupported split mode: {mode}")
        items = split_with_pattern(pattern)
        fallback = _normalize_form_token(cleaned)
        if items:
            return items, mode, 1.0 if len(items) > 1 else 0.7
        return ([fallback] if fallback else []), "none", 0.4

    best_mode = "none"
    best_items: list[str] = []
    for candidate_mode, pattern in _SPLIT_REGEX.items():
        items = split_with_pattern(pattern)
        if len(items) > len(best_items):
            best_mode = candidate_mode
            best_items = items
    if best_items:
        confidence = 0.95 if len(best_items) > 1 else 0.6
        return best_items, best_mode, confidence
    fallback = _normalize_form_token(cleaned)
    return ([fallback] if fallback else []), "none", 0.4


def _select_output_columns(df: pd.DataFrame, include_columns: list[str], exclude_columns: list[str]) -> pd.DataFrame:
    selected = df
    if include_columns:
        kept = [c for c in include_columns if c in selected.columns]
        selected = selected.loc[:, kept]
    if exclude_columns:
        to_drop = [c for c in exclude_columns if c in selected.columns]
        selected = selected.drop(columns=to_drop)
    return selected


def _split_dataframe_rows(
    df: pd.DataFrame,
    source_field: str,
    split_mode: str,
    custom_separator_regex: str | None,
    keep_unsplit: bool,
) -> pd.DataFrame:
    if source_field not in df.columns:
        raise KeyError(f"Column '{source_field}' not found. Available columns: {list(df.columns)}")
    rows: list[dict[str, str]] = []
    for idx, row in df.iterrows():
        raw_value = str(row[source_field])
        forms, split_rule, confidence = split_latin_forms(
            raw_value, mode=split_mode, custom_separator_regex=custom_separator_regex
        )
        if not forms and keep_unsplit:
            forms = [clean_anki_field(raw_value, truncate_at_block=False)]
        for form in forms:
            built = {column: str(row[column]) for column in df.columns}
            built["form"] = form
            built["form_norm"] = normalize_latin(form)
            built["source_field"] = source_field
            built["split_rule"] = split_rule
            built["split_confidence"] = f"{confidence:.2f}"
            built.setdefault("source_note_id", str(row.get("note_id", idx)))
            built.setdefault("source_notetype", str(row.get("note_type", "")))
            rows.append(built)
    return pd.DataFrame(rows)


def _parse_text_corpus(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            rows.append(
                {
                    "book": path.stem,
                    "chapter": 1,
                    "verse": idx,
                    "text": text,
                    "source_path": str(path),
                    "ref": f"{path.stem}:{idx}",
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["book", "chapter", "verse", "text", "text_norm", "source_path", "ref"])
    df["text_norm"] = df["text"].apply(normalize_latin)
    return df


def _parse_parallel_csv_corpus(path: Path, latin_column: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", keep_default_na=False)
    if latin_column not in df.columns:
        raise KeyError(f"Latin column '{latin_column}' not found in corpus CSV. Columns: {list(df.columns)}")
    out = pd.DataFrame()
    out["text"] = df[latin_column].astype(str)
    out["text_norm"] = out["text"].apply(normalize_latin)
    out["book"] = path.stem
    out["chapter"] = 1
    out["verse"] = range(1, len(out) + 1)
    out["source_path"] = str(path)
    out["ref"] = out["verse"].apply(lambda x: f"{path.stem}:{x}")
    for column in df.columns:
        if column == latin_column:
            continue
        out[f"translation_{column}"] = df[column].astype(str)
    return out


def load_corpora(
    paths: list[Path],
    corpus_format: str,
    latin_column: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        chosen_format = corpus_format
        if corpus_format == "auto":
            if path.suffix.lower() == ".xml":
                chosen_format = "usfx"
            elif path.suffix.lower() in {".csv", ".tsv"}:
                chosen_format = "csv"
            else:
                chosen_format = "txt"
        if chosen_format == "usfx":
            df = parse_usfx_to_df(path).copy()
            df["source_path"] = str(path)
            df["ref"] = df.apply(lambda r: f"{r['book']} {r['chapter']}:{r['verse']}", axis=1)
            frames.append(df)
        elif chosen_format == "csv":
            frames.append(_parse_parallel_csv_corpus(path, latin_column=latin_column))
        elif chosen_format == "txt":
            frames.append(_parse_text_corpus(path))
        else:
            raise ValueError(f"Unsupported corpus format: {chosen_format}")
    if not frames:
        raise ValueError("No corpus files provided.")
    merged = pd.concat(frames, ignore_index=True)
    if "text_norm" not in merged.columns:
        merged["text_norm"] = merged["text"].astype(str).apply(normalize_latin)
    return merged


def _build_word_frequency(df: pd.DataFrame) -> dict[str, int]:
    freq: dict[str, int] = defaultdict(int)
    for text in df["text_norm"].astype(str):
        for token in re.findall(r"[a-z]+", text):
            freq[token] += 1
    return freq


def score_cloze_difficulty(term_norm: str, sentence_norm: str, freq: dict[str, int]) -> tuple[float, str]:
    word_len = len(term_norm)
    sent_len = len(re.findall(r"[a-z]+", sentence_norm))
    token_freq = freq.get(term_norm, 1)
    rarity = 1.0 / max(token_freq, 1)
    score = word_len * 0.4 + sent_len * 0.3 + rarity * 10.0
    if score < 6.0:
        return score, "easy"
    if score < 10.0:
        return score, "medium"
    return score, "hard"


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _resolve_parallel_columns(
    corpus_df: pd.DataFrame,
    translation_langs: list[str],
    parallel_mode: str,
) -> list[str]:
    translation_columns = [c for c in corpus_df.columns if c.startswith("translation_")]
    if not translation_columns:
        return []
    if translation_langs:
        wanted = [f"translation_{lang}" for lang in translation_langs]
        return [c for c in wanted if c in translation_columns]
    if parallel_mode == "ignore":
        stderr_console.print("[yellow][WARN][/yellow] Parallel corpus columns detected but ignored by configuration.")
        return []
    if parallel_mode == "include":
        return translation_columns
    if _is_interactive():
        include = typer.confirm("Parallel translations found. Include translation columns in output?", default=False)
        if include:
            return translation_columns
    else:
        stderr_console.print(
            "[yellow][WARN][/yellow] Parallel corpus columns detected, "
            "but running non-interactively. Ignoring translations."
        )
    return []


def _ensure_latin_stanza_resources() -> None:
    """Best-effort download/availability check for CLTK's Latin Stanza models."""
    try:
        import stanza  # type: ignore[import-untyped]
    except ImportError:
        return

    try:
        stanza.download("la", package="ittb", processors="tokenize,pos,lemma", verbose=False)
    except Exception:
        # Keep this best-effort so offline use still gives a clear downstream error.
        return


def _build_analysis_candidates(
    form: str,
    lemma: str,
    upos: str,
    xpos: str,
    morph_features: str,
) -> list[dict[str, str]]:
    primary = {
        "lemma": lemma,
        "upos": upos,
        "xpos": xpos,
        "morph_features": morph_features,
        "source": "cltk",
    }
    candidates = [primary]
    form_norm = normalize_latin(form)
    lemma_norm = normalize_latin(lemma)
    if form_norm and form_norm != lemma_norm:
        candidates.append(
            {
                "lemma": form_norm,
                "upos": upos,
                "xpos": xpos,
                "morph_features": morph_features,
                "source": "surface",
            }
        )
    return candidates


def _extract_llm_choice(content: str, max_choice: int) -> int:
    choice_match = re.search(r'"choice"\s*:\s*(\d+)', content)
    if choice_match:
        choice = int(choice_match.group(1))
    else:
        num_match = re.search(r"\b(\d+)\b", content)
        if not num_match:
            raise ValueError("No numeric choice found in LLM output.")
        choice = int(num_match.group(1))
    if choice < 1 or choice > max_choice:
        raise ValueError(f"LLM choice {choice} out of range 1..{max_choice}")
    return choice - 1


def _query_ollama_choice(form: str, candidates: list[dict[str, str]], model: str, endpoint: str) -> int:
    endpoint_base = endpoint.rstrip("/")
    if not endpoint_base.startswith("http://") and not endpoint_base.startswith("https://"):
        endpoint_base = "http://" + endpoint_base
    prompt = (
        "Choose the best grammatical analysis for this Latin form.\n"
        f"Form: {form}\n"
        "Candidates:\n"
        + "\n".join(
            [
                (
                    f"{idx}. lemma={candidate['lemma']}; upos={candidate['upos']}; "
                    f"xpos={candidate['xpos']}; morph={candidate['morph_features']}; source={candidate['source']}"
                )
                for idx, candidate in enumerate(candidates, start=1)
            ]
        )
        + '\nReply with JSON only, like {"choice": 1}.'
    )
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": "Return only JSON with a numeric 'choice' field.",
            },
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0},
    }
    req = urlrequest.Request(
        url=f"{endpoint_base}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=20) as response:
            response_body = response.read().decode("utf-8")
    except (urlerror.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    data = json.loads(response_body)
    content = str(data.get("message", {}).get("content", "")).strip()
    if not content:
        raise RuntimeError("Ollama response missing message content.")
    return _extract_llm_choice(content, max_choice=len(candidates))


def _select_analysis_candidate(
    form: str,
    candidates: list[dict[str, str]],
    use_llm: bool,
    llm_provider: str,
    llm_model: str,
    llm_endpoint: str,
) -> tuple[dict[str, str], str]:
    if not candidates:
        raise ValueError("No analysis candidates available for disambiguation.")
    if not use_llm or len(candidates) == 1:
        return candidates[0], "ok"
    if llm_provider != "ollama":
        raise ValueError(f"Unsupported LLM provider '{llm_provider}'. Only 'ollama' is currently supported.")
    try:
        chosen_index = _query_ollama_choice(form=form, candidates=candidates, model=llm_model, endpoint=llm_endpoint)
        status = "ok-llm" if chosen_index != 0 else "ok"
        return candidates[chosen_index], status
    except Exception:
        return candidates[0], "ok-llm-fallback"


def annotate_with_cltk(
    df: pd.DataFrame,
    form_column: str,
    use_llm: bool = False,
    llm_provider: str = "ollama",
    llm_model: str = "ministral-3:8b",
    llm_endpoint: str = "http://localhost:11434",
) -> pd.DataFrame:
    if form_column not in df.columns:
        raise KeyError(f"Form column '{form_column}' not found. Available: {list(df.columns)}")

    try:
        from cltk import NLP  # type: ignore[import-untyped]
        from cltk.alphabet.processes import LatinNormalizeProcess  # type: ignore[import-untyped]
        from cltk.dependency.processes import LatinStanzaProcess  # type: ignore[import-untyped]
        from cltk.languages.pipelines import LatinPipeline  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - import guard only
        raise RuntimeError("CLTK is not installed. Install with `poetry add cltk`.") from exc

    _ensure_latin_stanza_resources()

    try:
        pipeline = LatinPipeline(processes=[LatinNormalizeProcess, LatinStanzaProcess])
        nlp = NLP(language="lat", custom_pipeline=pipeline, suppress_banner=True)
    except Exception as exc:  # pragma: no cover - runtime/model guard
        raise RuntimeError(
            "Could not initialize CLTK Latin pipeline. Ensure CLTK data/models are available in your environment."
        ) from exc

    lemmas: list[str] = []
    upos_values: list[str] = []
    xpos_values: list[str] = []
    feature_values: list[str] = []
    analysis_count: list[int] = []
    analysis_status: list[str] = []
    llm_fallback_warned = False

    for form in df[form_column].astype(str):
        text = form.strip()
        if not text:
            lemmas.append("")
            upos_values.append("")
            xpos_values.append("")
            feature_values.append("")
            analysis_count.append(0)
            analysis_status.append("empty")
            continue
        try:
            doc = nlp.analyze(text=text)
            words = list(getattr(doc, "words", []))
            if not words:
                lemmas.append("")
                upos_values.append("")
                xpos_values.append("")
                feature_values.append("")
                analysis_count.append(0)
                analysis_status.append("no-analysis")
                continue
            first = words[0]
            lemma = str(getattr(first, "lemma", "") or "")
            upos = str(getattr(first, "upos", "") or "")
            xpos = str(getattr(first, "xpos", "") or "")
            feats = getattr(first, "features", "")
            features_text = str(feats or "")
            candidates = _build_analysis_candidates(
                form=text,
                lemma=lemma,
                upos=upos,
                xpos=xpos,
                morph_features=features_text,
            )
            selected, status = _select_analysis_candidate(
                form=text,
                candidates=candidates,
                use_llm=use_llm,
                llm_provider=llm_provider,
                llm_model=llm_model,
                llm_endpoint=llm_endpoint,
            )
            if status == "ok-llm-fallback" and not llm_fallback_warned:
                stderr_console.print(
                    "[yellow][WARN][/yellow] LLM disambiguation failed for at least one row; "
                    "falling back to CLTK analysis."
                )
                llm_fallback_warned = True
            lemmas.append(selected["lemma"])
            upos_values.append(selected["upos"])
            xpos_values.append(selected["xpos"])
            feature_values.append(selected["morph_features"])
            analysis_count.append(len(candidates))
            analysis_status.append(status)
        except Exception:
            lemmas.append("")
            upos_values.append("")
            xpos_values.append("")
            feature_values.append("")
            analysis_count.append(0)
            analysis_status.append("error")

    out: pd.DataFrame = df.copy()
    out["lemma"] = lemmas
    out["upos"] = upos_values
    out["xpos"] = xpos_values
    out["morph_features"] = feature_values
    out["analysis_count"] = analysis_count
    out["analysis_status"] = analysis_status

    non_empty = any(bool(str(form).strip()) for form in df[form_column].astype(str))
    has_success = any(status == "ok" for status in analysis_status)
    if non_empty and not has_success:
        raise RuntimeError(
            "CLTK produced no successful analyses. Make sure Latin Stanza resources are installed, e.g. run: "
            "`poetry run python -c \"import stanza; stanza.download('la', package='ittb', "
            "processors='tokenize,pos,lemma')\"`"
        )

    return out


def _make_note_guid(seed: str) -> str:
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]


def rewrite_apkg_with_split_cards(
    apkg_input: Path,
    apkg_output: Path,
    source_field: str,
    split_mode: str,
    custom_separator_regex: str | None,
) -> None:
    """Keep original notes and append split-note copies with corresponding cards."""
    with zipfile.ZipFile(apkg_input, "r") as zf, tempfile.TemporaryDirectory() as td:
        sqlite_name, is_zstd = _pick_collection_database_name(zf.namelist())
        db_path = Path(td) / "collection.anki2"
        raw = zf.read(sqlite_name)
        if is_zstd:
            import zstandard

            raw = zstandard.ZstdDecompressor().decompress(raw, max_output_size=512 * 1024 * 1024)
        with open(db_path, "wb") as dst:
            dst.write(raw)

        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
        added_notes = 0
        added_cards = 0
        try:
            names_by_mid = _load_field_names_by_mid(con)
            max_note_id_raw = con.execute("SELECT COALESCE(MAX(id), 0) FROM notes").fetchone()
            max_card_id_raw = con.execute("SELECT COALESCE(MAX(id), 0) FROM cards").fetchone()
            next_note_id = int(max_note_id_raw[0]) + 1 if max_note_id_raw else 1
            next_card_id = int(max_card_id_raw[0]) + 1 if max_card_id_raw else 1

            notes = con.execute("SELECT * FROM notes ORDER BY id").fetchall()
            cards_by_note: dict[int, list[sqlite3.Row]] = defaultdict(list)
            for card_row in con.execute("SELECT * FROM cards ORDER BY id"):
                cards_by_note[int(card_row["nid"])].append(card_row)

            for note in notes:
                note_id = int(note["id"])
                note_mid = int(note["mid"])
                source_idx = _resolve_note_field_index(
                    con,
                    source_field,
                    fallback_to_front=False,
                    mid=note_mid,
                    field_names_by_mid=names_by_mid,
                )
                fields = str(note["flds"]).split("\x1f")
                if source_idx >= len(fields):
                    continue
                forms, _, _ = split_latin_forms(
                    fields[source_idx],
                    mode=split_mode,
                    custom_separator_regex=custom_separator_regex,
                )
                if len(forms) <= 1:
                    continue

                for form in forms:
                    if normalize_latin(form) == normalize_latin(
                        clean_anki_field(fields[source_idx], truncate_at_block=True)
                    ):
                        continue
                    cloned_fields = list(fields)
                    cloned_fields[source_idx] = form
                    now = int(time.time())
                    new_note_id = next_note_id
                    next_note_id += 1
                    note_values = (
                        new_note_id,
                        _make_note_guid(f"{note_id}:{form}:{new_note_id}"),
                        note_mid,
                        now,
                        int(note["usn"]),
                        str(note["tags"]),
                        "\x1f".join(cloned_fields),
                        form,
                        int(note["csum"]),
                        int(note["flags"]),
                        str(note["data"]),
                    )
                    con.execute(
                        "INSERT INTO notes (id, guid, mid, mod, usn, tags, flds, sfld, csum, flags, data) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        note_values,
                    )
                    added_notes += 1

                    for card in cards_by_note.get(note_id, []):
                        new_card_id = next_card_id
                        next_card_id += 1
                        card_values = (
                            new_card_id,
                            new_note_id,
                            int(card["did"]),
                            int(card["ord"]),
                            now,
                            int(card["usn"]),
                            int(card["type"]),
                            int(card["queue"]),
                            int(card["due"]),
                            int(card["ivl"]),
                            int(card["factor"]),
                            int(card["reps"]),
                            int(card["lapses"]),
                            int(card["left"]),
                            int(card["odue"]),
                            int(card["odid"]),
                            int(card["flags"]),
                            str(card["data"]),
                        )
                        con.execute(
                            (
                                "INSERT INTO cards (id, nid, did, ord, mod, usn, type, queue, due, ivl, factor, reps, "
                                "lapses, left, odue, odid, flags, data) "
                                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                            ),
                            card_values,
                        )
                        added_cards += 1

            con.commit()
        finally:
            con.close()

        updated_raw = db_path.read_bytes()
        if is_zstd:
            import zstandard

            updated_raw = zstandard.ZstdCompressor().compress(updated_raw)

        with zipfile.ZipFile(apkg_output, "w") as out_zip:
            for info_obj in zf.infolist():
                data = updated_raw if info_obj.filename == sqlite_name else zf.read(info_obj.filename)
                out_zip.writestr(info_obj, data)

    success(f"Added {added_notes} split notes and {added_cards} cards. Wrote: {apkg_output.resolve()}")


def _build_cloze_column(
    df: pd.DataFrame,
    bible_df: pd.DataFrame,
    bucket: dict[str, list[int]],
    front_col: str,
    max_examples: int,
    joiner: str,
    stopwords: set[str],
    word_forms: dict[str, list[str]],
    lemma_index: dict[str, set[str]],
    ignore_patterns: list[re.Pattern[str]],
    multi_cloze_per_verse: bool,
) -> list[str]:
    if front_col not in df.columns:
        raise KeyError(f"Column '{front_col}' not found. Available columns: {list(df.columns)}")

    cloze_col = []
    for _, row in df.iterrows():
        front_val = strip_anki_field(str(row[front_col]))
        if not front_val:
            cloze_col.append("")
            continue
        front_norm = normalize_latin(front_val)
        if front_norm in stopwords:
            cloze_col.append("")
            continue
        if any(p.search(front_norm) for p in ignore_patterns):
            cloze_col.append("")
            continue
        # Prefer lemma_index (PR #8), fall back to word_forms (PR #7), then single term
        lookup_terms = lemma_index.get(front_norm) or set(word_forms.get(front_norm, [front_val]))
        clozes: list[str] = []
        for form in lookup_terms:
            remaining = max_examples - len(clozes)
            if remaining <= 0:
                break
            clozes.extend(
                generate_clozes_for_word(
                    bible_df,
                    form,
                    bucket,
                    max_examples=remaining,
                    multi_cloze_per_verse=multi_cloze_per_verse,
                    lookup_terms={normalize_latin(form)},
                )
            )
        cloze_col.append(joiner.join(clozes))
    return cloze_col


def read_word_forms(path: Path) -> dict[str, list[str]]:
    """Read optional lemma→forms mapping.

    Format: one entry per line, comma-separated.
    Example: ``amo,amo,amas,amat``
    """
    mapping: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            parts = [p.strip() for p in t.split(",") if p.strip()]
            if not parts:
                continue
            lemma = normalize_latin(parts[0])
            forms = [normalize_latin(p) for p in parts]
            # preserve order while dropping duplicates
            unique_forms = list(dict.fromkeys(forms))
            mapping[lemma] = unique_forms
    return mapping


def update_csv_with_cloze(
    csv_input: Path,
    csv_output: Path,
    usfx_path: Path,
    front_col: str,
    new_field: str,
    max_examples: int = 2,
    joiner: str = "<br><br>",
    stopwords_path: Path | None = None,
    word_forms_path: Path | None = None,
    lemmas_path: Path | None = None,
    ignore_patterns: list[str] | None = None,
    multi_cloze_per_verse: bool = False,
    overwrite: bool = True,
) -> None:
    if csv_input.suffix.lower() in {".apkg", ".colpkg"}:
        _update_apkg_with_cloze(
            apkg_input=csv_input,
            apkg_output=csv_output,
            usfx_path=usfx_path,
            front_col=front_col,
            new_field=new_field,
            max_examples=max_examples,
            joiner=joiner,
            stopwords_path=stopwords_path,
            overwrite=overwrite,
        )
        return

    with stderr_console.status("Loading USFX corpus..."):
        bible_df = parse_usfx_to_df(usfx_path)
        bucket = build_bucket_index(bible_df)
    info(f"Loaded USFX corpus: {usfx_path}")

    with stderr_console.status("Reading input..."):
        df = _load_input_to_dataframe(csv_input, front_col)
    info(f"Reading input: {csv_input}")

    stopwords = read_stopwords(stopwords_path) if stopwords_path else set()
    word_forms = read_word_forms(word_forms_path) if word_forms_path else {}
    lemma_index = read_lemma_index(lemmas_path) if lemmas_path else {}
    compiled_ignore_patterns = compile_ignore_patterns(ignore_patterns or [])
    if stopwords:
        info(f"Loaded {len(stopwords)} stopwords.")
    if word_forms:
        info(f"Loaded word-form mappings for {len(word_forms)} lemmas.")
    if lemma_index:
        info(f"Loaded {len(lemma_index)} lemma/form entries.")
    if compiled_ignore_patterns:
        info(f"Loaded {len(compiled_ignore_patterns)} ignore patterns.")

    cloze_col = _build_cloze_column(
        df,
        bible_df,
        bucket,
        front_col=front_col,
        max_examples=max_examples,
        joiner=joiner,
        stopwords=stopwords,
        word_forms=word_forms,
        lemma_index=lemma_index,
        ignore_patterns=compiled_ignore_patterns,
        multi_cloze_per_verse=multi_cloze_per_verse,
    )

    if new_field not in df.columns:
        df[new_field] = cloze_col
    else:
        if overwrite:
            df[new_field] = cloze_col
        else:
            merged = []
            for old, new in zip(df[new_field].astype(str), cloze_col, strict=False):
                if old.strip() and new.strip():
                    merged.append(old.strip() + joiner + new.strip())
                else:
                    merged.append((old or "") + (new or ""))
            df[new_field] = merged

    df.to_csv(csv_output, index=False, encoding="utf-8")
    success(f"Wrote: {csv_output.resolve()}")


app = typer.Typer(
    help=(
        "Build Latin study cards from Anki decks: inspect deck structure, split multi-form "
        "cards, annotate grammar, and generate corpus-based cloze examples."
    ),
    context_settings={"help_option_names": ["-h", "--help"]},
)


def generate_impl(
    input: Annotated[
        Path,
        typer.Option(
            ...,
            help="Path to Anki CSV export or .apkg file",
            exists=True,
            readable=True,
        ),
    ],
    usfx: Annotated[
        Path,
        typer.Option(
            ...,
            help="Path to Latin USFX XML corpus (e.g., lat-clementine.usfx.xml)",
            exists=True,
            readable=True,
        ),
    ],
    output: Annotated[Path, typer.Option(..., help="Path for the updated CSV output")],
    anki_front: Annotated[
        str,
        typer.Option("--anki-front", help="Name of the 'Front' field to match notes for updates"),
    ] = "Front",
    new_field: Annotated[
        str,
        typer.Option("--new-field", help="Name of the field to write cloze examples into"),
    ] = "VulgataCloze",
    max_examples: Annotated[int, typer.Option("--max-examples", help="Max cloze examples per word")] = 2,
    joiner: Annotated[str, typer.Option(help="HTML separator for multiple examples")] = "<br><br>",
    stopwords: Annotated[
        Path | None,
        typer.Option(
            help="Optional path to a stopwords file (one word per line)",
        ),
    ] = None,
    word_forms: Annotated[
        Path | None,
        typer.Option(help="Optional path to lemma→forms mapping file (CSV-like: lemma,form1,form2,...)"),
    ] = None,
    lemmas: Annotated[
        Path | None,
        typer.Option(
            help="Optional path to lemma/form groups (comma or whitespace separated)",
        ),
    ] = None,
    ignore_pattern: Annotated[
        list[str] | None,
        typer.Option(help="Regex pattern(s) for normalized front values to ignore; can be repeated"),
    ] = None,
    multi_cloze_per_verse: Annotated[
        bool,
        typer.Option(
            "--multi-cloze-per-verse",
            help="Cloze every matching occurrence in a verse instead of just the first one",
        ),
    ] = False,
    append: Annotated[bool, typer.Option(help="Append to existing values instead of overwriting")] = False,
) -> None:
    """Update an Anki CSV or APKG file with cloze examples from a Latin USFX corpus."""
    update_csv_with_cloze(
        csv_input=input,
        csv_output=output,
        usfx_path=usfx,
        front_col=anki_front,
        new_field=new_field,
        max_examples=max_examples,
        joiner=joiner,
        stopwords_path=stopwords,
        word_forms_path=word_forms,
        lemmas_path=lemmas,
        ignore_patterns=ignore_pattern or [],
        multi_cloze_per_verse=multi_cloze_per_verse,
        overwrite=not append,
    )


def preview_impl(
    input: Annotated[
        Path,
        typer.Option(
            ...,
            help="Path to Anki CSV export or .apkg file",
            exists=True,
            readable=True,
        ),
    ],
    usfx: Annotated[
        Path,
        typer.Option(
            ...,
            help="Path to Latin USFX XML corpus (e.g., lat-clementine.usfx.xml)",
            exists=True,
            readable=True,
        ),
    ],
    anki_front: Annotated[
        str,
        typer.Option("--anki-front", help="Name of the 'Front' field to match notes for updates"),
    ] = "Front",
    max_examples: Annotated[int, typer.Option("--max-examples", help="Max cloze examples per word")] = 2,
    joiner: Annotated[str, typer.Option(help="HTML separator for multiple examples")] = "<br><br>",
    stopwords: Annotated[
        Path | None,
        typer.Option(
            help="Optional path to a stopwords file (one word per line)",
        ),
    ] = None,
    word_forms: Annotated[
        Path | None,
        typer.Option(help="Optional path to lemma→forms mapping file (CSV-like: lemma,form1,form2,...)"),
    ] = None,
    lemmas: Annotated[
        Path | None,
        typer.Option(
            help="Optional path to lemma/form groups (comma or whitespace separated)",
        ),
    ] = None,
    ignore_pattern: Annotated[
        list[str] | None,
        typer.Option(help="Regex pattern(s) for normalized front values to ignore; can be repeated"),
    ] = None,
    multi_cloze_per_verse: Annotated[
        bool,
        typer.Option(
            "--multi-cloze-per-verse",
            help="Cloze every matching occurrence in a verse instead of just the first one",
        ),
    ] = False,
    limit: Annotated[int, typer.Option("--limit", help="Max number of preview rows to print")] = 5,
) -> None:
    """Show a sample of generated clozes without writing output."""
    with stderr_console.status("Loading USFX corpus..."):
        bible_df = parse_usfx_to_df(usfx)
        bucket = build_bucket_index(bible_df)
    info(f"Loaded USFX corpus: {usfx}")

    with stderr_console.status("Reading input..."):
        df = _load_input_to_dataframe(input, anki_front)
    info(f"Reading input: {input}")

    stopwords_set = read_stopwords(stopwords) if stopwords else set()
    word_forms_map = read_word_forms(word_forms) if word_forms else {}
    lemma_index = read_lemma_index(lemmas) if lemmas else {}
    compiled_ignore_patterns = compile_ignore_patterns(ignore_pattern or [])
    if stopwords_set:
        info(f"Loaded {len(stopwords_set)} stopwords.")
    if word_forms_map:
        info(f"Loaded word-form mappings for {len(word_forms_map)} lemmas.")
    if lemma_index:
        info(f"Loaded {len(lemma_index)} lemma/form entries.")
    if compiled_ignore_patterns:
        info(f"Loaded {len(compiled_ignore_patterns)} ignore patterns.")

    cloze_col = _build_cloze_column(
        df,
        bible_df,
        bucket,
        front_col=anki_front,
        max_examples=max_examples,
        joiner=joiner,
        stopwords=stopwords_set,
        word_forms=word_forms_map,
        lemma_index=lemma_index,
        ignore_patterns=compiled_ignore_patterns,
        multi_cloze_per_verse=multi_cloze_per_verse,
    )

    table = Table(title="Cloze Preview", box=box.SIMPLE_HEAVY)
    table.add_column("Front", style="bold")
    table.add_column("Cloze Examples", overflow="fold")

    shown = 0
    for front_val, cloze_val in zip(df[anki_front].astype(str), cloze_col, strict=False):
        if not cloze_val.strip():
            continue
        table.add_row(Text(front_val), Text(cloze_val))
        shown += 1
        if shown >= limit:
            break

    if shown == 0:
        stdout_console.print("[yellow]No clozes generated for the provided input.[/yellow]")
    else:
        stdout_console.print(table)


def inspect_impl(
    input: Annotated[
        Path,
        typer.Option(
            ...,
            help="Path to Anki CSV export or .apkg/.colpkg file",
            exists=True,
            readable=True,
        ),
    ],
    notetype: Annotated[
        str | None,
        typer.Option(help="Optional note type filter for APKG inspection"),
    ] = None,
    fields: Annotated[
        list[str] | None,
        typer.Option(help="Field names to include in sample table; can be repeated"),
    ] = None,
    head: Annotated[int, typer.Option(help="Number of sample rows to display")] = 5,
) -> None:
    """Inspect deck schema and show a head-like sample preview."""
    suffix = input.suffix.lower()
    if suffix in {".apkg", ".colpkg"}:
        with zipfile.ZipFile(input, "r") as zf, tempfile.TemporaryDirectory() as td:
            sqlite_name, is_zstd = _pick_collection_database_name(zf.namelist())
            db_path = Path(td) / "collection.anki2"
            raw = zf.read(sqlite_name)
            if is_zstd:
                import zstandard

                raw = zstandard.ZstdDecompressor().decompress(raw, max_output_size=512 * 1024 * 1024)
            with open(db_path, "wb") as dst:
                dst.write(raw)

            con = sqlite3.connect(str(db_path))
            try:
                field_table = Table(title="Anki Note Types", box=box.SIMPLE_HEAVY)
                field_table.add_column("mid", justify="right")
                field_table.add_column("Note Type")
                field_table.add_column("Fields", overflow="fold")
                for mid, nt_name, nt_fields in _list_notetype_fields(con):
                    field_table.add_row(str(mid), nt_name, ", ".join(nt_fields))
                stdout_console.print(field_table)
            finally:
                con.close()

        df = _read_apkg_notes_dataframe(input, notetype=notetype)
    else:
        df = pd.read_csv(input, encoding="utf-8", keep_default_na=False)

    if df.empty:
        stdout_console.print("[yellow]No rows found in input.[/yellow]")
        return

    selected_fields = fields or []
    metadata_cols = [c for c in ["note_id", "note_type"] if c in df.columns]
    cols = metadata_cols + [c for c in selected_fields if c in df.columns] if selected_fields else list(df.columns)

    sample = df.loc[:, cols].head(head)
    sample_table = Table(title=f"Input Sample (first {len(sample)})", box=box.SIMPLE_HEAVY)
    for col in sample.columns:
        sample_table.add_column(col, overflow="fold")
    for _, row in sample.iterrows():
        sample_table.add_row(*[str(row[c]) for c in sample.columns])
    stdout_console.print(sample_table)

    if selected_fields:
        sep_table = Table(title="Separator Detection", box=box.SIMPLE_HEAVY)
        sep_table.add_column("Field")
        sep_table.add_column("comma", justify="right")
        sep_table.add_column("slash", justify="right")
        sep_table.add_column("pipe", justify="right")
        sep_table.add_column("semicolon", justify="right")
        for field_name in selected_fields:
            if field_name not in df.columns:
                continue
            values = df[field_name].astype(str)
            counts = {
                "comma": int(values.str.contains(",", regex=False).sum()),
                "slash": int(values.str.contains("/", regex=False).sum()),
                "pipe": int(values.str.contains("|", regex=False).sum()),
                "semicolon": int(values.str.contains(";", regex=False).sum()),
            }
            sep_table.add_row(
                field_name,
                str(counts["comma"]),
                str(counts["slash"]),
                str(counts["pipe"]),
                str(counts["semicolon"]),
            )
        stdout_console.print(sep_table)


def split_impl(
    input: Annotated[
        Path,
        typer.Option(
            ...,
            help="Path to Anki CSV export or .apkg/.colpkg file",
            exists=True,
            readable=True,
        ),
    ],
    output: Annotated[Path, typer.Option(..., help="Output file path (CSV by default)")],
    source_field: Annotated[str, typer.Option(help="Field that contains multiple forms")],
    notetype: Annotated[
        str | None,
        typer.Option(help="Optional APKG note type filter"),
    ] = None,
    split_mode: Annotated[
        str,
        typer.Option(help="Split mode: auto, comma, slash, pipe, semicolon, custom"),
    ] = "auto",
    custom_separator_regex: Annotated[
        str | None,
        typer.Option(help="Regex separator when --split-mode=custom"),
    ] = None,
    keep_unsplit: Annotated[
        bool,
        typer.Option(help="Keep rows where no multi-form separator is detected"),
    ] = False,
    include_column: Annotated[
        list[str] | None,
        typer.Option(help="Output columns to include; can be repeated"),
    ] = None,
    exclude_column: Annotated[
        list[str] | None,
        typer.Option(help="Output columns to exclude; can be repeated"),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(help="Output format: csv (default) or apkg"),
    ] = "csv",
) -> None:
    """Split multi-form cards into one-row-per-form records."""
    if output_format == "apkg":
        if input.suffix.lower() not in {".apkg", ".colpkg"}:
            raise ValueError("APKG rewrite mode requires --input to be .apkg or .colpkg")
        rewrite_apkg_with_split_cards(
            apkg_input=input,
            apkg_output=output,
            source_field=source_field,
            split_mode=split_mode,
            custom_separator_regex=custom_separator_regex,
        )
        return

    df = _load_any_input_dataframe(input, notetype=notetype)
    split_df = _split_dataframe_rows(
        df,
        source_field=source_field,
        split_mode=split_mode,
        custom_separator_regex=custom_separator_regex,
        keep_unsplit=keep_unsplit,
    )
    final_df = _select_output_columns(split_df, include_column or [], exclude_column or [])
    final_df.to_csv(output, index=False, encoding="utf-8")
    success(f"Wrote split output: {output.resolve()} ({len(final_df)} rows)")


def annotate_impl(
    input: Annotated[
        Path,
        typer.Option(..., help="Path to CSV with a form column", exists=True, readable=True),
    ],
    output: Annotated[Path, typer.Option(..., help="Output CSV path")],
    form_column: Annotated[str, typer.Option(help="Column name containing Latin forms")] = "form",
    include_column: Annotated[
        list[str] | None,
        typer.Option(help="Output columns to include; can be repeated"),
    ] = None,
    exclude_column: Annotated[
        list[str] | None,
        typer.Option(help="Output columns to exclude; can be repeated"),
    ] = None,
    use_llm: Annotated[
        bool,
        typer.Option(help="Enable optional LLM-assisted disambiguation (fallback remains CLTK-only)"),
    ] = False,
    llm_provider: Annotated[
        str,
        typer.Option(help="LLM provider when --use-llm is enabled"),
    ] = "ollama",
    llm_model: Annotated[
        str,
        typer.Option(help="LLM model name when --use-llm is enabled"),
    ] = "ministral-3:8b",
    llm_endpoint: Annotated[
        str,
        typer.Option(help="Ollama base URL when --use-llm is enabled"),
    ] = os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
) -> None:
    """Annotate CSV forms with CLTK lemma/POS/morphology metadata."""
    if use_llm:
        info(f"LLM disambiguation enabled: provider={llm_provider}, model={llm_model}, endpoint={llm_endpoint}")
        stderr_console.print(
            "[yellow][WARN][/yellow] LLM disambiguation fallback is currently experimental; "
            "using CLTK output as source of truth."
        )
    df = pd.read_csv(input, encoding="utf-8", keep_default_na=False)
    annotated = annotate_with_cltk(
        df,
        form_column=form_column,
        use_llm=use_llm,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_endpoint=llm_endpoint,
    )
    final_df = _select_output_columns(annotated, include_column or [], exclude_column or [])
    final_df.to_csv(output, index=False, encoding="utf-8")
    success(f"Wrote annotated output: {output.resolve()} ({len(final_df)} rows)")


def cloze_impl(
    input: Annotated[
        Path,
        typer.Option(..., help="Path to CSV with a form column", exists=True, readable=True),
    ],
    output: Annotated[Path, typer.Option(..., help="Output CSV path")],
    corpus: Annotated[
        list[Path],
        typer.Option(
            ...,
            help="One or more corpus paths (repeatable): USFX XML, text files, or parallel CSV",
            exists=True,
            readable=True,
        ),
    ],
    form_column: Annotated[str, typer.Option(help="Input column containing the target form")] = "form",
    cloze_column: Annotated[str, typer.Option(help="Output column for generated clozes")] = "CorpusCloze",
    corpus_format: Annotated[
        str,
        typer.Option(help="Corpus format: auto, usfx, txt, csv"),
    ] = "auto",
    latin_column: Annotated[
        str,
        typer.Option(help="Latin column when using corpus CSV input"),
    ] = "la",
    max_examples: Annotated[int, typer.Option(help="Max cloze examples per form")] = 2,
    joiner: Annotated[str, typer.Option(help="HTML separator for multiple examples")] = "<br><br>",
    multi_cloze_per_verse: Annotated[
        bool,
        typer.Option("--multi-cloze-per-verse", help="Cloze all matches in a sentence instead of first only"),
    ] = False,
    difficulty: Annotated[
        str,
        typer.Option(help="Difficulty filter: any, easy, medium, hard"),
    ] = "any",
    parallel_mode: Annotated[
        str,
        typer.Option(help="Parallel corpus behavior when translation columns are discovered: prompt, include, ignore"),
    ] = "prompt",
    translation_lang: Annotated[
        list[str] | None,
        typer.Option(help="Translation languages to include (e.g. en, de); can be repeated"),
    ] = None,
    include_column: Annotated[
        list[str] | None,
        typer.Option(help="Output columns to include; can be repeated"),
    ] = None,
    exclude_column: Annotated[
        list[str] | None,
        typer.Option(help="Output columns to exclude; can be repeated"),
    ] = None,
) -> None:
    """Generate corpus-based cloze cards for each form in a CSV input."""
    df = pd.read_csv(input, encoding="utf-8", keep_default_na=False)
    if form_column not in df.columns:
        raise KeyError(f"Form column '{form_column}' not found. Available: {list(df.columns)}")

    corpus_df = load_corpora(corpus, corpus_format=corpus_format, latin_column=latin_column)
    bucket = build_bucket_index(corpus_df)
    freq = _build_word_frequency(corpus_df)
    translation_columns = _resolve_parallel_columns(
        corpus_df,
        translation_langs=translation_lang or [],
        parallel_mode=parallel_mode,
    )

    cloze_values: list[str] = []
    difficulty_scores: list[str] = []
    difficulty_labels: list[str] = []
    translation_values: dict[str, list[str]] = {column: [] for column in translation_columns}

    for form in df[form_column].astype(str):
        matches = collect_cloze_matches(
            corpus_df,
            word=form,
            bucket=bucket,
            max_examples=max_examples,
            multi_cloze_per_verse=multi_cloze_per_verse,
            translation_columns=translation_columns,
        )
        if not matches:
            cloze_values.append("")
            difficulty_scores.append("")
            difficulty_labels.append("")
            for column in translation_columns:
                translation_values[column].append("")
            continue

        rendered = [f"{m['cloze']} <span style='color:#888'>({m['ref']})</span>" for m in matches]
        cloze_values.append(joiner.join(rendered))

        scores = [score_cloze_difficulty(normalize_latin(form), m["sentence_norm"], freq)[0] for m in matches]
        labels = [score_cloze_difficulty(normalize_latin(form), m["sentence_norm"], freq)[1] for m in matches]
        avg_score = sum(scores) / len(scores)
        if avg_score < 6.0:
            label = "easy"
        elif avg_score < 10.0:
            label = "medium"
        else:
            label = "hard"
        if labels and label not in {"easy", "medium", "hard"}:
            label = labels[0]
        difficulty_scores.append(f"{avg_score:.2f}")
        difficulty_labels.append(label)
        for column in translation_columns:
            joined = " | ".join(dict.fromkeys(m.get(column, "") for m in matches if m.get(column, "")))
            translation_values[column].append(joined)

    out = df.copy()
    out[cloze_column] = cloze_values
    out["difficulty_score"] = difficulty_scores
    out["difficulty_label"] = difficulty_labels
    for column, values in translation_values.items():
        out[column] = values

    if difficulty != "any":
        out = out[out["difficulty_label"] == difficulty]

    final_df = _select_output_columns(out, include_column or [], exclude_column or [])
    final_df.to_csv(output, index=False, encoding="utf-8")
    success(f"Wrote cloze output: {output.resolve()} ({len(final_df)} rows)")


def validate_impl(
    input: Annotated[
        Path,
        typer.Option(
            ...,
            help="Path to Anki CSV export or .apkg file",
            exists=True,
            readable=True,
        ),
    ],
    usfx: Annotated[
        Path,
        typer.Option(
            ...,
            help="Path to Latin USFX XML corpus (e.g., lat-clementine.usfx.xml)",
            exists=True,
            readable=True,
        ),
    ],
    anki_front: Annotated[
        str,
        typer.Option("--anki-front", help="Name of the 'Front' field expected in the input file"),
    ] = "Front",
) -> None:
    """Validate USFX parsing integrity and required input columns."""
    checks: list[dict[str, str]] = []
    all_ok = True

    # USFX integrity checks
    try:
        bible_df = parse_usfx_to_df(usfx)
        expected_usfx_columns = ["book", "chapter", "verse", "text", "text_norm"]
        missing_usfx_columns = [c for c in expected_usfx_columns if c not in bible_df.columns]

        if missing_usfx_columns:
            all_ok = False
            checks.append(
                {
                    "name": "USFX columns",
                    "status": _status_label(False),
                    "details": f"Missing parsed columns: {missing_usfx_columns}",
                }
            )
        else:
            checks.append(
                {
                    "name": "USFX columns",
                    "status": _status_label(True),
                    "details": "All expected verse columns are present.",
                }
            )

        empty_verse_rows = int((bible_df["text"].astype(str).str.strip() == "").sum())
        verses_count = len(bible_df)
        if empty_verse_rows > 0:
            all_ok = False
            checks.append(
                {
                    "name": "USFX verse text",
                    "status": _status_label(False),
                    "details": f"Found {empty_verse_rows} verse rows with empty text.",
                }
            )
        else:
            checks.append(
                {
                    "name": "USFX verse text",
                    "status": _status_label(True),
                    "details": f"Parsed {verses_count} verses with non-empty text.",
                }
            )
    except (ET.ParseError, ValueError) as exc:
        all_ok = False
        checks.append(
            {
                "name": "USFX structure",
                "status": _status_label(False),
                "details": f"Could not parse USFX structure: {exc}",
            }
        )

    # Input integrity checks
    input_df: pd.DataFrame | None = None
    try:
        input_df = _load_input_to_dataframe(input, anki_front)
    except (ValueError, csv.Error) as exc:
        # Validation should still check columns if this is a CSV with ambiguous
        # header detection from csv.Sniffer in the strict loader.
        if input.suffix.lower() not in {".apkg", ".colpkg"}:
            try:
                input_df = pd.read_csv(input, encoding="utf-8", keep_default_na=False)
                checks.append(
                    {
                        "name": "Input parser",
                        "status": _status_label(True),
                        "details": (
                            "CSV header detection was ambiguous for the strict loader, "
                            "but validation recovered using a direct CSV read."
                        ),
                    }
                )
            except Exception:
                input_df = None

        if input_df is None:
            all_ok = False
            checks.append(
                {
                    "name": "Input file",
                    "status": _status_label(False),
                    "details": f"Could not parse input data: {exc}",
                }
            )

    if input_df is not None:
        if anki_front not in input_df.columns:
            all_ok = False
            checks.append(
                {
                    "name": "Input columns",
                    "status": _status_label(False),
                    "details": f"Missing required column '{anki_front}'. Available: {list(input_df.columns)}",
                }
            )
        else:
            non_empty_front = int((input_df[anki_front].astype(str).str.strip() != "").sum())
            checks.append(
                {
                    "name": "Input columns",
                    "status": _status_label(True),
                    "details": (f"Column '{anki_front}' is present. Non-empty rows: {non_empty_front}/{len(input_df)}"),
                }
            )

    _render_validation_table(checks)

    if all_ok:
        success("Validation passed: all integrity checks succeeded.")
        return

    stderr_console.print("[bold red][ERROR][/bold red] Validation failed. See report above.")
    raise typer.Exit(code=1)


def _register_commands() -> None:
    from .commands.annotate import annotate
    from .commands.cloze import cloze
    from .commands.generate import generate
    from .commands.inspect import inspect
    from .commands.preview import preview
    from .commands.split import split
    from .commands.validate import validate

    app.command()(generate)
    app.command()(preview)
    app.command()(inspect)
    app.command()(split)
    app.command()(annotate)
    app.command()(cloze)
    app.command()(validate)


_register_commands()
