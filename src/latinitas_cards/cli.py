import csv
import re
import sys
import xml.etree.ElementTree as ET
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
    rows = []
    current_book = None
    current_chapter = None

    for elem in root.iter():
        tag = elem.tag.lower()
        if tag in ("book", "b"):
            current_book = elem.attrib.get("id") or elem.attrib.get("code") or elem.attrib.get("n")
        elif tag in ("c", "chapter"):
            val = elem.attrib.get("id") or elem.attrib.get("n") or "0"
            m = re.search(r"\d+", str(val))
            current_chapter = int(m.group()) if m else 0
        elif tag in ("v", "verse"):
            val = elem.attrib.get("id") or elem.attrib.get("n") or "0"
            m = re.search(r"\d+", str(val))
            verse_num = int(m.group()) if m else 0
            text = "".join(elem.itertext()).strip()
            if text:
                rows.append({"book": current_book, "chapter": current_chapter, "verse": verse_num, "text": text})

    df = pd.DataFrame(rows).dropna(subset=["book", "chapter", "verse"])
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

    eprint(f"[INFO] Reading Anki CSV: {csv_input}")
    with open(csv_input, newline="", encoding="utf-8") as f:
        sample = f.read(4096)
        dialect = csv.Sniffer().sniff(sample)
        has_header = csv.Sniffer().has_header(sample)

    if not has_header:
        raise ValueError("CSV seems to have no header row. Please export with headers (Front, Back, ...).")

    df = pd.read_csv(csv_input, encoding="utf-8", dialect=dialect, keep_default_na=False)

    if front_col not in df.columns:
        raise KeyError(f"Column '{front_col}' not found in CSV. Available columns: {list(df.columns)}")

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
            help="Path to Anki CSV export (with header, e.g., Front,Back,...)",
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
        typer.Option("--anki-front", help="Name of the 'Front' column to match notes for updates"),
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
    """Update an Anki CSV file with cloze examples from the Latin Vulgate."""
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
