# Principal-part parser support matrix

The principal-part parser consumes a `CanonicalSourceRecord` and its confirmed
`DeckProfile`. It does not infer a profile from source text, choose a recipe, or
render card content. The profile's role order and separator values are the
semantic contract for that source.

## Supported matrix

| Area | Supported contract | Not inferred |
| --- | --- | --- |
| Semantic roles | Any confirmed profile with at least two distinct, non-empty role names. The generic default is `present_1s`, `present_infinitive`, `perfect_1s`, and `supine`; the approved sanitized-fixture profile explicitly uses `perfect_passive_participle` (PPP) in slot four. | A position is never renamed from its text, and role names are never guessed from a lemma or ending. |
| Complete layout | Exactly one segment for every configured role. | Extra segments are not silently discarded. |
| Deponent/exceptional layout | A profile may explicitly confirm a shorter layout, such as `present_1s`, `present_infinitive`, `perfect_1s` for `sequor — sequi — secutus sum`. The parser treats the third value as the confirmed third role; it does not detect “deponent” itself. | A four-role profile receiving three unmarked values is not treated as a deponent entry. It is ambiguous because the missing role is not identified. |
| Reusable separator | A single configured literal separator is reused between every role, for example `(" — ",)` or `(", ")`. | Commas, semicolons, slashes, dashes, and newlines are not accepted as alternatives unless confirmed in the profile. |
| Position-specific separators | Exactly `role_count - 1` configured literal separators, in order, for example `(", ", "; ", " / ")`. | The parser does not reorder or substitute separators. |
| Explicit omission | A blank delimited slot is preserved as an omitted role, for example `sum — esse — fui — `. Non-leading omissions return success with `display=None` and `comparison=None`. | The first two roles are required; an explicitly blank leading role returns `incomplete`. |

Display values are trimmed only at delimiter boundaries. Internal spacing,
capitalization, Unicode, macrons, and other display content are preserved. The
comparison value is separate: it uses case-folding, Unicode decomposition with
combining marks removed, and whitespace collapsing. Comparison normalization is
never used to rewrite display content.

`ParsedPrincipalParts.semantic_roles` and `identity_roles` retain the confirmed
role names and order. Downstream identity code may use those semantic role keys;
it must not substitute comparison text or positional guesses.

The generic `supine` default is retained for legacy/profile compatibility. It is not
interchangeable with the approved sanitized-fixture role
`perfect_passive_participle` (Partizip Perfekt Passiv, PPP). Role order is changed only
explicitly during profile confirmation; unsafe source or exercise assignments remain
structured skip/review outcomes. Structural parser counts do not claim universal
morphological coverage or prove eligible-verb coverage.

## Structured outcomes

| Status | Example failed assumption | Meaning |
| --- | --- | --- |
| `success` | The confirmed fields, separators, and role count agree. | `PrincipalPartParseSuccess.value` contains named `PrincipalPartValue` objects. |
| `incomplete` | Empty field, too few segments, missing position-specific separator, or omitted leading role. | The source does not provide enough data for a supported interpretation. |
| `unsupported` | The source uses commas while the confirmed profile requires an em dash. | The source layout is different from the confirmed profile and needs confirmation rather than fallback parsing. |
| `ambiguous` | A reusable separator produces an unmarked short layout or an extra separator. | More than one semantic interpretation remains; no role assignment is returned. |

Every failure includes a stable `code`, a human-readable `assumption`, and
source identity/location when the canonical record provides them. Failure
messages do not echo principal-part field contents.

These are parser fixtures and profile-driven parser contracts. T-011 validated the
representative-deck mapping and initial German terminology recorded in
`docs/representative-deck-validation.md`; this does not turn the parser into a
universal morphological analyzer.
