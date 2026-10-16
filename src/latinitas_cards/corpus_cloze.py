"""Project normalized JSONL passages into legacy cloze contexts."""

from pathlib import Path

import pandas as pd

from .cli_contract import Code, ContractError
from .corpus_io import CorpusFinding, read_corpus
from .corpus_model import CorpusPassage
from .usfx_adapter import normalize_latin


def load_jsonl_corpus(path: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for record in read_corpus(path):
        if isinstance(record, CorpusFinding):
            raise ContractError(Code.INVALID_RECORD, f"Line {record.line}: {record.errors}", path=str(path))
        if isinstance(record, CorpusPassage):
            rows.append(
                {
                    "text": record.text,
                    "text_norm": normalize_latin(record.text),
                    "display_ref": record.citation.display,
                    "source_path": str(path),
                    **{f"translation_{item.language}": item.text for item in record.translations or []},
                }
            )
    return pd.DataFrame(rows, columns=None if rows else ["text", "text_norm", "display_ref", "source_path"]).fillna("")
