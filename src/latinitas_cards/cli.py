import csv
import re
import sqlite3
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Annotated

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
    if not s:
        return s
    # Truncate at first block-level tag (supplementary hint content)
    m = _BLOCK_TAG_RE.search(s)
    if m:
        s = s[: m.start()]
    # Replace HTML entities
    for entity, char in _HTML_ENTITY_MAP.items():
        s = s.replace(entity, char)
    # Strip remaining inline HTML tags
    s = _HTML_TAG_RE.sub("", s)
    # Decompose Unicode and drop combining diacritical marks (macrons, breves, etc.)
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
    for idx, row in df.iterrows():
        tn = row["text_norm"]
        if not tn:
            continue
        letters = set(re.findall(r"[a-z]", tn[:60])) or {"*"}
        for ch in letters:
            bucket[ch].append(idx)  # type: ignore[arg-type]
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


def generate_clozes_for_word(
    df: pd.DataFrame, word: str, bucket: dict[str, list[int]], max_examples: int = 2
) -> list[str]:
    patt = make_word_regex(word)
    word_norm = normalize_latin(word)
    out = []
    cnt = 0
    for idx in candidate_indices(word_norm, bucket, len(df)):
        verse_text = str(df.at[idx, "text"])
        verse_norm = str(df.at[idx, "text_norm"])
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


def _read_apkg_field_rows(apkg_path: Path, field_name: str) -> list[dict[str, str]]:
    """
    Read notes from an .apkg/.colpkg and extract a single field by name,
    returning rows like [{'Front': '...'}, ...].

    Supports both plain SQLite (.anki2/.anki21) and zstd-compressed (.anki21b)
    collection databases.  When a .anki21b is present it is preferred over the
    plain .anki2 (which modern Anki exports populate with only a placeholder note).
    """
    rows: list[dict[str, str]] = []

    # Extract the SQLite DB to a temp file
    with zipfile.ZipFile(apkg_path, "r") as zf, tempfile.TemporaryDirectory() as td:
        names = zf.namelist()

        # Prefer zstd-compressed anki21b when available
        sqlite_name: str | None = None
        is_zstd = False
        for name in names:
            if name.endswith(".anki21b"):
                sqlite_name = name
                is_zstd = True
                break
        if not sqlite_name:
            for name in names:
                if name.endswith(".anki2") or name.endswith(".anki21"):
                    sqlite_name = name
                    break
        if not sqlite_name:
            raise ValueError("APKG does not contain a collection database (.anki2/.anki21/.anki21b).")

        db_path = Path(td) / "collection.anki2"
        with zf.open(sqlite_name) as src:
            raw = src.read()
        if is_zstd:
            import zstandard

            raw = zstandard.ZstdDecompressor().decompress(raw, max_output_size=256 * 1024 * 1024)
        with open(db_path, "wb") as dst:
            dst.write(raw)

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
        return pd.read_csv(  # type: ignore[call-overload,no-any-return]
            input_path, encoding="utf-8", dialect=dialect, keep_default_na=False
        )


def _build_cloze_column(
    df: pd.DataFrame,
    bible_df: pd.DataFrame,
    bucket: dict[str, list[int]],
    front_col: str,
    max_examples: int,
    joiner: str,
    stopwords: set[str],
) -> list[str]:
    if front_col not in df.columns:
        raise KeyError(f"Column '{front_col}' not found. Available columns: {list(df.columns)}")

    cloze_col = []
    for _, row in df.iterrows():
        front_val = strip_anki_field(str(row[front_col]))
        if not front_val:
            cloze_col.append("")
            continue
        if normalize_latin(front_val) in stopwords:
            cloze_col.append("")
            continue
        clozes = generate_clozes_for_word(bible_df, front_val, bucket, max_examples=max_examples)
        cloze_col.append(joiner.join(clozes))
    return cloze_col


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
) -> None:
    with stderr_console.status("Loading Vulgata USFX..."):
        bible_df = parse_usfx_to_df(usfx_path)
        bucket = build_bucket_index(bible_df)
    info(f"Loaded Vulgata USFX: {usfx_path}")

    with stderr_console.status("Reading input..."):
        df = _load_input_to_dataframe(csv_input, front_col)
    info(f"Reading input: {csv_input}")

    stopwords = read_stopwords(stopwords_path) if stopwords_path else set()
    if stopwords:
        info(f"Loaded {len(stopwords)} stopwords.")

    cloze_col = _build_cloze_column(
        df,
        bible_df,
        bucket,
        front_col=front_col,
        max_examples=max_examples,
        joiner=joiner,
        stopwords=stopwords,
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
    help="Generate Anki cloze examples from the Latin Vulgate and update your Anki CSV export.",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command()
def generate(
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
) -> None:
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


@app.command()
def preview(
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
    limit: Annotated[int, typer.Option("--limit", help="Max number of preview rows to print")] = 5,
) -> None:
    """Show a sample of generated clozes without writing output."""
    with stderr_console.status("Loading Vulgata USFX..."):
        bible_df = parse_usfx_to_df(usfx)
        bucket = build_bucket_index(bible_df)
    info(f"Loaded Vulgata USFX: {usfx}")

    with stderr_console.status("Reading input..."):
        df = _load_input_to_dataframe(input, anki_front)
    info(f"Reading input: {input}")

    stopwords_set = read_stopwords(stopwords) if stopwords else set()
    if stopwords_set:
        info(f"Loaded {len(stopwords_set)} stopwords.")

    cloze_col = _build_cloze_column(
        df,
        bible_df,
        bucket,
        front_col=anki_front,
        max_examples=max_examples,
        joiner=joiner,
        stopwords=stopwords_set,
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


@app.command()
def validate(
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
