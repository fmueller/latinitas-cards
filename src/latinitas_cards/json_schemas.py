"""Registry of new versioned JSON document schemas."""

import json

from pydantic import BaseModel

from .annotation_plans import AnnotationPatch, AnnotationPlan, AnnotationReceipt
from .annotation_sidecar import AnnotationSchema
from .cli_contract import CliResult
from .corpus_manifest import CorpusManifest
from .corpus_model import CorpusDescriptor, CorpusRecordSchema
from .policy import Policy

SCHEMAS: dict[str, type[BaseModel]] = {
    "latinitas-annotation-patch": AnnotationPatch,
    "latinitas-annotation-plan": AnnotationPlan,
    "latinitas-annotation-receipt": AnnotationReceipt,
    "latinitas-annotations": AnnotationSchema,
    "latinitas-cli-result": CliResult,
    "latinitas-corpus": CorpusRecordSchema,
    "latinitas-corpus-descriptor": CorpusDescriptor,
    "latinitas-corpus-manifest": CorpusManifest,
    "latinitas-policy": Policy,
}


def render_schema(model: type[BaseModel]) -> str:
    return json.dumps(model.model_json_schema(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
