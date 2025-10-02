import csv
import re
import sqlite3
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def normalize_latin(s: str) -> str:
    if s is None:
        return ""
    s = s.lower().strip()
    s = s.replace("æ", "ae").replace("œ", "oe")
    s = s.replace("j", "i")
    return s


def make_word_regex(word: str) -> re.Pattern:
    w = re.escape(normalize_latin(word))
    return re.compile(rf"\b{w}\b", flags=re.IGNORECASE)


def cloze_once(text: str, pattern: re.Pattern):
    def repl(m):
        return "{{c1::" + m.group(0) + "}}"

    new_text, n = pattern.subn(repl, text, count=1)
    return new_text, n


def parse_usfx_to_df(path: Path):
    tree = ET.parse(path)
    root = tree.getroot()

    rows: list[dict] = []

    current_book: str | None = None
    current_chapter: int | None = None
    current_verse: int | None = None
    buffer: list[str] = []

    number_re = re.compile(r"\d+")

    def norm_space(s: str) -> str:
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def flush():
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
    return df.sort_values(["book", "chapter", "verse"]).reset_index(drop=True)


def build_bucket_index(df):
    bucket = defaultdict(list)
    for idx, row in df.iterrows():
        tn = row["text_norm"]
        if not tn:
            continue
        letters = set(re.findall(r"[a-z]", tn[:60])) or {"*"}
        for ch in letters:
            bucket[ch].append(idx)
    return bucket


def candidate_indices(word_norm, bucket, total_len):
    first = next((c for c in word_norm if c.isalpha()), "*")
    return bucket.get(first, range(total_len))


def read_stopwords(path: Path) -> set:
    stops = set()
    if not path:
        return stops
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            stops.add(normalize_latin(t))
    return stops


def generate_clozes_for_word(df, word, bucket, max_examples=2):
    patt = make_word_regex(word)
    word_norm = normalize_latin(word)
    out = []
    cnt = 0
    for idx in candidate_indices(word_norm, bucket, len(df)):
        verse_text = df.at[idx, "text"]
        verse_norm = df.at[idx, "text_norm"]
        if word_norm not in verse_norm:
            continue
        cloze, n = cloze_once(verse_text, patt)
        if n > 0:
            ref = f"{df.at[idx, 'book']} {df.at[idx, 'chapter']}:{df.at[idx, 'verse']}"
            out.append(f"{cloze} <span style='color:#888'>({ref})</span>")
            cnt += 1
            if cnt >= max_examples:
                break
    return out


def _read_apkg_field_rows(apkg_path: Path, field_name: str) -> list[dict]:
    """
    Read notes from an .apkg and extract a single field by name, returning rows like [{'Front': '...'}, ...].
    Assumptions:
      - .colpkg is a zip with collection.anki2 (SQLite)
      - notes.flds holds all field values separated by \x1f (unit separator)
      - field order comes from the model JSON inside col.models; we use the first model’s field order if present,
        else assume first field is 'Front'.
    """
    rows: list[dict] = []

    # Extract the SQLite DB to a temp file
    with zipfile.ZipFile(apkg_path, "r") as zf, tempfile.TemporaryDirectory() as td:
        # Usually it is collection.anki2, sometimes called collection.anki21
        sqlite_name = None
        for name in zf.namelist():
            if name.endswith(".anki2") or name.endswith(".anki21"):
                sqlite_name = name
                break
        if not sqlite_name:
            raise ValueError("APKG does not contain a collection .anki2/.anki21 database.")

        db_path = Path(td) / "collection.anki2"
        with zf.open(sqlite_name) as src, open(db_path, "wb") as dst:
            dst.write(src.read())

        # Connect to the DB
        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
        try:
            # Discover field names order from the first model if available
            # col table has a single row with JSON meta; but to keep it robust without external JSON libs,
            # we fallback to generic mapping if JSON parsing is not strict here.
            # We instead infer field count from the first note and assume 'Front' maps to index 0 if not found.
            cur = con.execute("SELECT flds FROM notes LIMIT 1")
            first = cur.fetchone()
            if not first:
                return rows

            # Default: if front-like field exists in common positions
            candidate_index = 0  # fallback to index 0
            front_like_names = ["Front", "front", "Expression", "Word"]
            # Try to infer from notetypes if available
            try:
                # Read models JSON from col table (optional best-effort)
                meta = con.execute("SELECT models FROM col LIMIT 1").fetchone()
                if meta and meta[0]:
                    import json

                    models = json.loads(meta[0])
                    # Pick first model, find fields order
                    if isinstance(models, dict) and models:
                        first_model = next(iter(models.values()))
                        if isinstance(first_model, dict) and "flds" in first_model:
                            names = [f.get("name", "") for f in first_model["flds"]]
                            # Find requested field name (case-insensitive)
                            lowered = [n.lower() for n in names]
                            if field_name.lower() in lowered:
                                candidate_index = lowered.index(field_name.lower())
                            else:
                                # try front-like defaults
                                for n in front_like_names:
                                    if n.lower() in lowered:
                                        candidate_index = lowered.index(n.lower())
                                        break
                        # else fallback to first
            except Exception:
                # Best-effort; ignore JSON parsing failures
                pass

            # If still not found, try to guess by common names embedded in first note (no labels available),
            # else stick to index 0
            # Now iterate all notes
            cur = con.execute("SELECT flds FROM notes")
            for r in cur:
                parts = r["flds"].split("\x1f")
                if candidate_index < len(parts):
                    rows.append({field_name: parts[candidate_index]})
                else:
                    # skip malformed notes
                    continue
            return rows
        finally:
            con.close()


def _load_input_to_dataframe(input_path: Path, front_col: str) -> pd.DataFrame:
    """
    Load either CSV (with header) or APKG into a DataFrame exposing the front_col.
    """
    suffix = input_path.suffix.lower()
    if suffix in {".colpkg"}:
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
        return pd.read_csv(input_path, encoding="utf-8", dialect=dialect, keep_default_na=False)


def update_csv_with_cloze(
    csv_input: Path,
    csv_output: Path,
    usfx_path: Path,
    front_col: str,
    new_field: str,
    max_examples: int = 2,
    joiner: str = "<br><br>",
    stopwords_path: Path | None = None,
    overwrite: bool = True,
):
    eprint(f"[INFO] Loading Vulgata USFX: {usfx_path}")
    bible_df = parse_usfx_to_df(usfx_path)
    bucket = build_bucket_index(bible_df)

    eprint(f"[INFO] Reading input: {csv_input}")
    df = _load_input_to_dataframe(csv_input, front_col)

    if front_col not in df.columns:
        raise KeyError(f"Column '{front_col}' not found. Available columns: {list(df.columns)}")

    stopwords = read_stopwords(stopwords_path) if stopwords_path else set()
    if stopwords:
        eprint(f"[INFO] Loaded {len(stopwords)} stopwords.")

    cloze_col = []
    for _, row in df.iterrows():
        front_val = str(row[front_col]).strip()
        if not front_val:
            cloze_col.append("")
            continue
        if normalize_latin(front_val) in stopwords:
            cloze_col.append("")
            continue
        clozes = generate_clozes_for_word(bible_df, front_val, bucket, max_examples=max_examples)
        cloze_col.append(joiner.join(clozes))

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
    eprint(f"[OK] Wrote: {csv_output.resolve()}")


app = typer.Typer(help="Generate Anki cloze examples from the Latin Vulgate and update your Anki CSV export.")


@app.command()
def vulgata_cloze_cli(
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
            help="Path to Vulgate USFX XML (e.g., lat-clementine.usfx.xml)",
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
    append: Annotated[bool, typer.Option(help="Append to existing values instead of overwriting")] = False,
):
    """Update an Anki CSV or APKG file with cloze examples from the Latin Vulgate."""
    update_csv_with_cloze(
        csv_input=input,
        csv_output=output,
        usfx_path=usfx,
        front_col=anki_front,
        new_field=new_field,
        max_examples=max_examples,
        joiner=joiner,
        stopwords_path=stopwords,
        overwrite=not append,
    )
