---
id: T-136-reject-segment-aliases-to-resolved-default
title: Reject segment aliases to resolved default-manifest DATA locks
status: todo
priority: high
spec_ref: specs/v0.3.0.md#sidecar-annotation-model
dependencies: []
updated_at: "2026-10-16T20:58:27Z"
---

# T-136-reject-segment-aliases-to-resolved-default Reject segment aliases to resolved default-manifest DATA locks

## Description

Final E2E Round 5 at `ee6cdb97510169db89e193a02f13aacce4434fcb` confirms
segmentation misses the resolved lock of a symlinked default-manifest DATA file.
The Sidecar Annotation Model requires writer aliases to manifests/locks to be
rejected. Normalizing through this explicit default-manifest symlink succeeds;
no arbitrary custom-manifest discovery is involved.

Severity medium: reserved-path alias acceptance and forbidden file creation.
Priority high follows the round's filing instruction. Held reserved bytes/inode
and normalizer exclusion remain intact; no concurrent corruption or detached
held-lock claim is made.

Dedupe: explicit reopening overlap with completed T-131/T-129 and T-113. The
original 126-case segment matrix still passes; the newly exercised default
manifest DATA symlink exposes an omitted resolved reservation. T-134 reports
the related apply path, but segment has its own writer input-protection list.
T-133 is apply-only, T-130/T-132 claims-only, and future T-128 unrelated. This
standalone task has no dependencies or follow-up to T-124 or the other findings.

### Self-contained reproduction

Run from the checkout. Every path is disposable and the manifest is the known
default, not a custom unknown path:

```sh
root=$(mktemp -d)
cli="$PWD/.venv/bin/latinitas-cards"
printf 'Canis currit.\nFemina librum legit.\n' > "$root/source.txt"
"$cli" corpus normalize "$root/source.txt" --adapter text --corpus-id segment --output "$root/corpus.jsonl" --json
mv "$root/corpus.manifest.json" "$root/manifest-data.json"
ln -s manifest-data.json "$root/corpus.manifest.json"
"$cli" corpus normalize "$root/source.txt" --adapter text --corpus-id segment --output "$root/corpus.jsonl" --json
rm "$root/manifest-data.json.lock"
ln -s manifest-data.json.lock "$root/hop"
ln -s hop "$root/side.jsonl.lock"
"$cli" annotations segment "$root/corpus.jsonl" --output "$root/side.jsonl" --timestamp 2026-01-02T03:04:05Z --json
test -e "$root/manifest-data.json.lock" && echo 'Unexpected reserved file created'
wc -l "$root/side.jsonl"
rm -rf "$root"
```

Expected segment exit 3/alias, absent reserved lock and absent sidecar. Actual
exit 0/appended 2, newly created zero-byte reserved lock and a three-line
sidecar. Existing output-hardlink variant exits 0/appended 2 under a real held
manifest lock. Existing output-lock symlink gives 4/busy held and 0/appended 2
released rather than 3/alias in both cases. Ordinary output succeeds.

## Acceptance

- Reject output/output-lock aliases to the resolved default-manifest DATA lock,
  including existing held/released and dangling paths, with 3/alias/no write.
- Keep original T-129/T-131 matrix, normalizer exclusion and ordinary controls
  passing; preserve reserved bytes/inodes and absent reservations.
- Preserve existing sidecar behavior and exact logical claims locks. No unknown
  manifest discovery, product redesign or future T-128 work is implied.

## Verification Notes

- Round 5 bundle `round5-evidence.tar.gz` includes the independently authored
  `round5/coordinator/segment-manifest-replay.py`, exact generated inputs,
  before/after hashes, safe environment, DEVNULL, timeouts and raw CLI streams.
  Fresh replay: 20 CLI calls, 36 assertions, 14 failures. All positive controls,
  held inode/bytes and real normalizer exclusion/retry assertions pass.
- Source check: `commands/annotations.py` passes only the logical default
  manifest lock to `append_records`; normalization resolves manifest DATA
  before constructing its acquired lock. This is separate from apply's list.

## Implementation Notes

Unstarted. Finding-only delivery; no product fix or unit-suite substitution.
