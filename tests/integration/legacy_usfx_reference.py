"""Frozen verbatim pre-T-108 parser and normalizer; test-only differential oracle."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


def normalize_latin(s: str) -> str:
    if s is None:
        return ""
    s = s.lower().strip()
    s = s.replace("æ", "ae").replace("œ", "oe")
    s = s.replace("j", "i")
    return s


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
    return df.sort_values(["book", "chapter", "verse"]).reset_index(drop=True)
