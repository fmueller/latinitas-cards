"""Optional Chromium contract check using public exports, not synthetic wrappers."""

import csv
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from phrase_inputs_test import call, saved_example
from phrase_recipes_test import exercise

from latinitas_cards.phrase_recipes import PHRASE_LAYOUT
from latinitas_cards.reference_templates import REFERENCE_CARD_CSS


@pytest.mark.skipif(shutil.which("agent-browser") is None, reason="Chromium/agent-browser is optional")
def test_exported_phrase_panels_follow_reference_appearance(tmp_path: Path) -> None:
    source, selections = saved_example(tmp_path)
    candidate = call("phrases", "generate", str(source), str(selections))["candidates"][0]
    recipes = tmp_path / "recipes.json"
    recipes.write_text(
        json.dumps(
            {
                "version": 1,
                "max_exercises": 2,
                "exercises": [
                    exercise(recipe)
                    .model_copy(update={"candidate_key": candidate["candidate_key"]})
                    .model_dump(mode="json")
                    for recipe in ("phrase_translation", "phrase_recognition")
                ],
            }
        )
    )
    args = [str(source), "--selections", str(selections), "--recipes", str(recipes)]
    for claim in call("claims", "list", *args)["claims"]:
        call(
            "claims",
            "record",
            *args,
            "--fingerprint",
            claim["fingerprint"],
            "--status",
            "accepted",
            "--reviewer",
            "fixture teacher",
            "--reason",
            "Individually checked reference",
        )
    profile_path = tmp_path / "profile.json"
    profile = json.loads(profile_path.read_text())
    session = "phrase-appearance"

    def browser(*arguments: str) -> str:
        return subprocess.check_output(["agent-browser", "--session", session, *arguments], text=True)

    try:
        for theme in ("muted", "monochrome"):
            for appearance in (None, "light", "dark", "auto"):
                profile["morphology"]["theme"] = theme
                if appearance is None:
                    profile["morphology"].pop("appearance", None)
                else:
                    profile["morphology"]["appearance"] = appearance
                profile_path.write_text(json.dumps(profile))
                exported = tmp_path / f"{theme}-{appearance or 'default'}.csv"
                call("phrases", "export", str(source), str(selections), str(recipes), str(exported))
                lines = exported.read_text().splitlines()
                columns = next(
                    line.removeprefix("#columns:").split(",") for line in lines if line.startswith("#columns:")
                )
                rows = list(csv.DictReader((line for line in lines if not line.startswith("#")), fieldnames=columns))
                for template in PHRASE_LAYOUT[10:]:
                    row = next(row for row in rows if row[template.slot.enabled_field])
                    for markup in (template.front, template.back):
                        markup = re.sub(r"\{\{[#/][^}]+\}\}", "", markup)
                        for name, value in row.items():
                            markup = markup.replace("{{" + name + "}}", value)
                        page = tmp_path / "card.html"
                        page.write_text(
                            '<!doctype html><meta charset="utf-8"><style>'
                            + REFERENCE_CARD_CSS
                            + '</style><body class="card">'
                            + markup
                            + "</body>"
                        )
                        browser("open", page.as_uri())
                        browser("set", "viewport", "390", "844", "2")
                        for night in (False, True, False):
                            client_class = "card nightMode" if night else "card"
                            actual = json.loads(
                                browser(
                                    "eval",
                                    """(() => {
                                document.body.className = CLIENT_CLASS;
                                const panel = document.body.firstElementChild;
                                const style = getComputedStyle(panel);
                                return [style.color, style.backgroundColor,
                                    style.getPropertyValue('--stem').trim(), document.scripts.length,
                                    document.documentElement.scrollWidth > innerWidth];
                            })()""".replace("CLIENT_CLASS", json.dumps(client_class)),
                                )
                            )
                            dark = appearance == "dark" or appearance == "auto" and night
                            colors = (
                                ["rgb(238, 238, 238)", "rgb(34, 37, 38)"]
                                if dark
                                else ["rgb(34, 34, 34)", "rgb(247, 247, 244)"]
                            )
                            stem = "currentColor" if theme == "monochrome" else "#9bc9b9" if dark else "#356357"
                            assert actual == [*colors, stem, 0, False], (theme, appearance, template.slot.recipe, night)
                        # Removing the real exported panel classes must defeat the check.
                        control = json.loads(
                            browser(
                                "eval",
                                """(() => {
                            document.body.firstElementChild.className = '';
                            const style = getComputedStyle(document.body.firstElementChild);
                            return [style.color, style.backgroundColor];
                        })()""",
                            )
                        )
                        assert control != colors
    finally:
        browser("close")
