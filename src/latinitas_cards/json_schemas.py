"""Registry of new versioned JSON document schemas."""

import json

from pydantic import BaseModel

from .cli_contract import CliResult
from .corpus_model import CorpusDescriptor, CorpusRecordSchema

SCHEMAS: dict[str, type[BaseModel]] = {
    "latinitas-cli-result": CliResult,
    "latinitas-corpus": CorpusRecordSchema,
    "latinitas-corpus-descriptor": CorpusDescriptor,
}


def render_schema(model: type[BaseModel]) -> str:
    return json.dumps(model.model_json_schema(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
