"""Corpus exercise candidates and their identities for the four corpus recipes.

Candidates are generated per occurrence from the accepted-fact view, eligible or not.
Identity digests the recipe, target language, unit and folded target only, so review
state, fact values and profile settings never change an exercise's ID. Rendering,
ranking and selection consume these candidates later.
"""

import hashlib
from dataclasses import dataclass
from typing import Literal

from .canonical import canonical_json, digest
from .corpus_facts import FactReason, FactView, PassageFacts, Resolution, Sentence, Span
from .corpus_model import latin_fold
from .corpus_profile import CorpusProfile, CorpusRecipe
from .identity import derive_card_semantic_key, derive_latinitas_id

type CandidateReason = FactReason | Literal["invalid_span"]
type UnitKey = dict[str, str | int]
type TargetKey = dict[str, str | int]

SOURCE_SCOPE = "corpus-passage-v1"
OBJECT_KEY_PREFIX = "corpus-exercise-v1-"
# Fixed Contracts table order, also the corpus-v1 slot order.
RECIPE_ORDER: tuple[CorpusRecipe, ...] = (
    "corpus_lexical_recognition",
    "corpus_guided_analysis",
    "corpus_phrase_translation",
    "corpus_sentence_translation",
)
PHRASE_WORDS = range(2, 9)


@dataclass(frozen=True)
class Candidate:
    """One exercise the profile's recipes could produce; ``reasons`` is empty when eligible.

    An ``invalid_span`` phrase candidate has no unit, target or identity, only its
    ``annotation_id``. ``filtered`` marks a fragment unit that guided analysis and sentence
    translation never use.
    """

    recipe: CorpusRecipe
    passage_id: str
    unit: UnitKey | None
    target: TargetKey | None
    span: Span | None
    reasons: tuple[CandidateReason, ...]
    resolutions: tuple[Resolution, ...] = ()
    object_key: str | None = None
    latinitas_id: str | None = None
    card_key: str | None = None
    context_digest: str | None = None
    filtered: Literal["filtered:fragment"] | None = None
    annotation_id: str | None = None


def exercise_identity(
    passage_id: str, recipe: CorpusRecipe, target_language: str, unit: UnitKey, target: TargetKey | None
) -> tuple[str, str]:
    """Return ``(object_key, LatinitasID)``; only the lowercased primary subtag is identity."""
    payload = {
        "recipe": recipe,
        "target_language": _primary_subtag(target_language),
        "unit": unit,
        "target": target,
    }
    object_key = OBJECT_KEY_PREFIX + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return object_key, derive_latinitas_id(passage_id, object_key, source_scope=SOURCE_SCOPE)


def context_digest(unit_text: str, target_text: str | None) -> str:
    """Structured digest of the folded unit and target texts, stored outside identity."""
    return digest(
        {"unit_text": latin_fold(unit_text), "target_text": None if target_text is None else latin_fold(target_text)}
    )


def exercise_candidates(view: FactView, profile: CorpusProfile) -> tuple[Candidate, ...]:
    """Candidates of the profile's recipes in passage order, then recipe and unit order."""
    language = _primary_subtag(profile.target_language)
    if _primary_subtag(view.target_language) != language:
        raise ValueError(f"fact view target language {view.target_language!r} differs from the profile")
    recipes = {entry.recipe for entry in profile.recipes}
    result: list[Candidate] = []
    for passage in sorted(view.passages.values(), key=lambda facts: facts.passage.order):
        # Without a current segmentation a passage has no units, and its token facts cannot resolve.
        if passage.segmentation is not None:
            builder = _Passage(passage, language)
            for recipe in RECIPE_ORDER:
                if recipe in recipes:
                    result.extend(builder.candidates(recipe, profile.limits.max_tokens))
    return tuple(result)


class _Passage:
    def __init__(self, facts: PassageFacts, language: str) -> None:
        assert facts.segmentation is not None
        self.facts = facts
        self.language = language
        self.text = facts.passage.text
        self.tokens: list[Span] = [(token.span[0], token.span[1]) for token in facts.segmentation.tokens]
        self.words = [index for index, token in enumerate(facts.segmentation.tokens) if token.kind == "word"]

    def candidates(self, recipe: CorpusRecipe, max_tokens: int) -> list[Candidate]:
        if recipe == "corpus_phrase_translation":
            return self._phrases()
        if recipe == "corpus_sentence_translation":
            return self._translations(max_tokens)
        return [
            candidate
            for sentence in self.facts.sentences
            for position, token in enumerate(self._sentence_words(sentence))
            if (candidate := self._token(recipe, sentence, position, token)) is not None
        ]

    def _token(self, recipe: CorpusRecipe, sentence: Sentence, position: int, token: int) -> Candidate | None:
        span = self.tokens[token]
        if ("morphology", span) not in self.facts.facts:
            return None
        forms = self._forms(sentence)
        target: TargetKey = {"form": forms[position], "occurrence": forms[:position].count(forms[position])}
        if recipe == "corpus_lexical_recognition":
            resolutions = (self.facts.morphology(token), self.facts.translation(span))
            return self._candidate(recipe, sentence, target, span, resolutions)
        resolutions = (self.facts.analysis(token), self.facts.sentence_translation(sentence.index))
        return self._candidate(recipe, sentence, target, span, resolutions, fragment_filtered=True)

    def _phrases(self) -> list[Candidate]:
        sentence_spans = {sentence.span for sentence in self.facts.sentences}
        word_spans = {self.tokens[token] for token in self.words}
        result: list[Candidate] = []
        for (layer, span), facts in sorted(self.facts.facts.items(), key=lambda item: item[0][1] or (-1, -1)):
            if layer != "translation" or span is None or span in sentence_spans or span in word_spans:
                continue
            window = self._window(span)
            if window is None:
                result.extend(
                    Candidate(
                        recipe="corpus_phrase_translation",
                        passage_id=self.facts.passage.passage_id,
                        unit=None,
                        target=None,
                        span=span,
                        reasons=("invalid_span",),
                        annotation_id=fact.annotation_id,
                    )
                    for fact in facts
                )
                continue
            sentence, first, size = window
            forms = self._forms(sentence)
            windows = [" ".join(forms[start : start + size]) for start in range(first + 1)]
            target: TargetKey = {"form": windows[-1], "occurrence": windows[:-1].count(windows[-1])}
            resolutions = (self.facts.translation(span),)
            result.append(self._candidate("corpus_phrase_translation", sentence, target, span, resolutions))
        return result

    def _window(self, span: Span) -> tuple[Sentence, int, int] | None:
        """A qualifying span: 2-8 word tokens, from a word's start to a word's end, a proper sub-span."""
        for sentence in self.facts.sentences:
            words = [self.tokens[token] for token in self._sentence_words(sentence)]
            starts = [position for position, word in enumerate(words) if word[0] == span[0]]
            ends = [position for position, word in enumerate(words) if word[1] == span[1]]
            if starts and ends:
                size = ends[0] - starts[0] + 1
                if size in PHRASE_WORDS and size < len(words):
                    return sentence, starts[0], size
                return None
        return None

    def _translations(self, max_tokens: int) -> list[Candidate]:
        result: list[Candidate] = []
        sentences = self.facts.sentences
        for sentence in sentences:
            keys = [("translation", sentence.span)] + ([("translation", None)] if len(sentences) == 1 else [])
            if any(key in self.facts.facts for key in keys):
                resolutions = (self.facts.sentence_translation(sentence.index),)
                result.append(
                    self._candidate(
                        "corpus_sentence_translation", sentence, None, None, resolutions, fragment_filtered=True
                    )
                )
        if len(sentences) >= 2 and ("translation", None) in self.facts.facts and len(self.words) <= max_tokens:
            unit: UnitKey = {"kind": "passage"}
            resolutions = (self.facts.passage_translation(),)
            result.append(self._build("corpus_sentence_translation", unit, self.text, None, None, resolutions))
        return result

    def _candidate(
        self,
        recipe: CorpusRecipe,
        sentence: Sentence,
        target: TargetKey | None,
        span: Span | None,
        resolutions: tuple[Resolution, ...],
        fragment_filtered: bool = False,
    ) -> Candidate:
        unit: UnitKey = {"kind": "sentence", "index": sentence.index}
        unit_text = self.text[sentence.span[0] : sentence.span[1]]
        filtered: Literal["filtered:fragment"] | None = (
            "filtered:fragment" if fragment_filtered and sentence.fragment else None
        )
        return self._build(recipe, unit, unit_text, target, span, resolutions, filtered)

    def _build(
        self,
        recipe: CorpusRecipe,
        unit: UnitKey,
        unit_text: str,
        target: TargetKey | None,
        span: Span | None,
        resolutions: tuple[Resolution, ...],
        filtered: Literal["filtered:fragment"] | None = None,
    ) -> Candidate:
        passage_id = self.facts.passage.passage_id
        object_key, latinitas_id = exercise_identity(passage_id, recipe, self.language, unit, target)
        reasons = tuple(dict.fromkeys(r.reason for r in resolutions if r.reason is not None))
        return Candidate(
            recipe=recipe,
            passage_id=passage_id,
            unit=unit,
            target=target,
            span=span,
            reasons=reasons,
            resolutions=resolutions,
            object_key=object_key,
            latinitas_id=latinitas_id,
            card_key=derive_card_semantic_key(recipe, "context"),
            context_digest=context_digest(unit_text, None if span is None else self.text[span[0] : span[1]]),
            filtered=filtered,
        )

    def _sentence_words(self, sentence: Sentence) -> list[int]:
        return [token for token in self.words if sentence.tokens[0] <= token < sentence.tokens[1]]

    def _forms(self, sentence: Sentence) -> list[str]:
        return [latin_fold(self.text[slice(*self.tokens[token])]) for token in self._sentence_words(sentence)]


def _primary_subtag(language: str) -> str:
    return language.lower().split("-")[0]
