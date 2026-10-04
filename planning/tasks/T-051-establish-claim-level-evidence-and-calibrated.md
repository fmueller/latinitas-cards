---
id: T-051-establish-claim-level-evidence-and-calibrated
title: Establish claim-level evidence and calibrated review gates
status: todo
priority: high
spec_ref: specs/v0.2.0.md#calibrated-form-parsing
dependencies:
    - T-049-establish-reviewed-source-extraction-fixtures-and
updated_at: "2026-10-04T23:05:46Z"
---

# T-051-establish-claim-level-evidence-and-calibrated Establish claim-level evidence and calibrated review gates

## Description

Define and implement the precision-first claim/review contract independently of extraction and display styling. Establish reviewed domain-relevant evaluation before automatic acceptance.

## Acceptance

- Represent individual grammatical labels, segmentation, and explanation claims with evidence/provenance, alternatives, acceptance or withholding status, and reasons. A whole-analysis confidence score is insufficient.
- Establish a reviewed evaluation sample, independently authored expected claims, measurable precision, and explicit project-policy thresholds; document applicability and limitations. Do not invent threshold values without evaluation.
- Declare development and evaluation material separately or explicitly disclose overlap. Report accepted correct/error claims, accepted coverage, and withheld cases per relevant claim/analyzer category, with sample limits for each threshold decision. Extraction counts and principal-part results do not calibrate later token parsing.
- Evaluate automatic claims by relevant claim/analyzer category. Until evidence supports a threshold, withhold automatic acceptance and require explicit review.
- Expose a review path for accepting or withholding individual claims with recorded decisions bound to the specific candidate/claim and evidence/profile version. Changed evidence or configuration requires revalidation or renewed review, not silent reuse. A setup-proposed fourth-role label is not an explicitly reviewed PPP/supine choice. Knowledge acceptance does not approve destination writes.
- Analyzer output, LLM self-confidence, successful extraction, and unreviewed examples never establish calibration. Optional LLM use remains network-explicit; core operation is offline.
- Tests distinguish accepted, rejected, ambiguous, and unsupported claims and ensure unsupported segmentation/explanations cannot leak into asserted knowledge.

## Verification

Publish the evaluation results and policy decision. For code, use red/green tests and run the mandatory ruff, mypy, pytest -v chain.
