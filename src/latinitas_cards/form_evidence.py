"""Finite component and context evidence read independently of form proposals."""

import re

from .acceptance_records import canonical
from .claim_review import Evidence
from .latin_library import attested_paradigms, bennett_sections
from .phrase_inputs import Features


def attested_segmentation(
    lemma: str, pos: str, surface: str, features: Features
) -> tuple[dict[str, object] | None, tuple[Evidence, ...]]:
    for paradigm in attested_paradigms():
        if paradigm.lemma != lemma or paradigm.pos != pos:
            continue
        if not any(c.surface == surface and c.features == features for c in paradigm.cells):
            continue
        if pos == "noun":
            references = bennett_sections(17, 20)
            stem = references[1].text.split("stem, ", 1)[1].split("-", 1)[0]
            role, convention = "case-ending", "bennett-declensional-stem/1"
        elif pos == "verb":
            references = bennett_sections(96, 97, 117)
            stem = paradigm.anchors[1].surface[:-2]
            role, convention = "personal-ending", "bennett-present-stem/1"
        else:
            continue
        if not surface.startswith(stem) or surface == stem:
            continue
        ending = surface[len(stem) :]
        if pos == "verb":
            rows = references[0].text.split("Sing. " if features.number == "singular" else "Plu. ", 1)[1].splitlines()
            row = rows.index(str(features.person) + ".")
            if ending != rows[row + 1].split(";", 1)[0].removeprefix("-"):
                continue
        return {
            "convention": convention,
            "components": [{"role": "stem", "text": stem}, {"role": role, "text": ending}],
            "operation": "concatenation",
            "surface": surface,
        }, (*references, paradigm.evidence)
    return None, ()


def contextual_ppp(
    lemma: str, surface: str, context: str, *, cloze: bool
) -> tuple[Features | None, tuple[Evidence, ...]]:
    """Recognize exact attested perfect-passive compounds, not arbitrary prose.

    Nominal features belong to the participle, not to the finite compound. The
    context's auxiliary supplies the subject person/number and predicate role.
    """
    references = bennett_sections(99, 100, 101, 102, 63, 95, 166, 233, 234, 254)
    principal_parts = references[2].text.split("PERF. PASS. PARTIC.\n", 1)[1].splitlines()[:4]
    if lemma != principal_parts[0]:
        return None, ()
    if cloze:
        if context.count("____") != 1 or surface in context:
            return None, ()
        restored = context.replace("____", surface)
        start = context.index("____")
    else:
        if context.count(surface) != 1:
            return None, ()
        restored, start = context, context.index(surface)
    rows = references[3].text.split("\nPERFECT\n", 1)[1].split("\nPLUPERFECT.", 1)[0].splitlines()[1:]
    auxiliaries = references[1].text.split("PRESENT TENSE.\nSINGULAR.\nPLURAL.\n", 1)[1].splitlines()[:6]
    for index, row in enumerate(rows):
        compound = re.sub(r" \([^)]*\)|\[\d+\]", "", row)
        participle, auxiliary = compound.split()
        if compound != restored or participle != surface or start != 0:
            continue
        if auxiliary != auxiliaries[index].split(",", 1)[0]:
            continue
        number = "singular" if index % 2 == 0 else "plural"
        nominal_table = references[4].text
        if number == "plural":
            nominal_table = nominal_table.split("PLURAL.\n", 1)[1]
        nominal_row = nominal_table.split("Nom.\n", 1)[1].splitlines()[:3]
        endings = [word.removeprefix("bon") for word in nominal_row]
        variants = re.search(r"\(([^)]*)\)", rows[index % 2])
        if variants is None or variants[1].replace("-", "").split(", ") != endings[1:]:
            continue
        if not participle.endswith(endings[0]):
            continue
        features = Features.model_validate(
            dict(case="nominative", number=number, gender="masculine", tense="perfect", voice="passive")
        )
        binding = Evidence(
            "attested-context-binding",
            references[3].reference + "/indicative/perfect/" + str(index),
            references[3].version,
            canonical(
                {
                    "context": restored,
                    "target_span": [start, start + len(surface)],
                    "role": "perfect-passive-compound-predicate-participle",
                    "features": features,
                    "auxiliary": {
                        "surface": auxiliary,
                        "person": str(index // 2 + 1),
                        "number": number,
                        "tense": "present",
                        "mood": "indicative",
                        "voice": "active",
                    },
                    "compound": {"tense": "perfect", "mood": "indicative", "voice": "passive"},
                    "nominal_basis": {
                        "classification": "adjective-form",
                        "construction": "subject-agreement",
                        "case": "nominative",
                        "gender": "masculine",
                        "number": number,
                        "paradigm_exemplar": nominal_row[0],
                        "rules": ["95", "166", "233.3", "234", "254.1", "254.2"],
                    },
                }
            ).decode(),
        )
        return features, (*references, binding)
    return None, ()
