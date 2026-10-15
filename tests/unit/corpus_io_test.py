import json
import runpy
import sys
from pathlib import Path

import pytest

from latinitas_cards.atomic_io import writer_lock
from latinitas_cards.canonical import canonical_json, digest
from latinitas_cards.cli_contract import Code, ContractError
from latinitas_cards.corpus_io import CorpusFinding, read_corpus, write_corpus
from latinitas_cards.corpus_model import (
    CorpusHeader,
    CorpusMetadata,
    CorpusPassage,
    CorpusRecord,
    CorpusWork,
    record_digest,
)

FIXTURES = Path(__file__).parents[1] / "fixtures/corpus"


def header() -> CorpusHeader:
    return CorpusHeader.model_validate(
        {
            "format": "latinitas-corpus",
            "version": 1,
            "corpus_id": "test-corpus",
            "adapter": "text/1",
            "source": {"file_name": "source.txt", "sha256": "sha256:" + "a" * 64, "bytes": 5},
        }
    )


def test_reader_streams_and_yields_findings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "input.jsonl"
    path.write_text(
        canonical_json({**header().model_dump(), "corpus_id": "BAD"})
        + '\n{"record":"work","work":"GEN","order":1.0}\n{"record":"other"}\n'
    )

    def no_bulk(*args: object, **kwargs: object) -> None:
        pytest.fail("reader must stream")

    monkeypatch.setattr(Path, "read_text", no_bulk)
    monkeypatch.setattr(Path, "read_bytes", no_bulk)
    items = list(read_corpus(path))
    assert [item.line for item in items if isinstance(item, CorpusFinding)] == [1, 2, 3]
    assert isinstance(items[0], CorpusFinding) and items[0].raw["corpus_id"] == "BAD"
    assert isinstance(items[1], CorpusFinding) and items[1].code == "float_value"


@pytest.mark.parametrize(
    "content,code",
    [
        ("", Code.MISSING_HEADER),
        ("[]\n", Code.INVALID_JSON),
        ("oops\n", Code.INVALID_JSON),
        ('{"record":"work"}\n', Code.MISSING_HEADER),
        ('{"record":"header","format":"other","version":1}\n', Code.UNSUPPORTED_FORMAT),
        *[
            (
                json.dumps({"record": "header", "format": "latinitas-corpus", "version": v}) + "\n",
                Code.UNSUPPORTED_VERSION,
            )
            for v in [2, 1.0, True, "1", None]
        ],
    ],
)
def test_gate_errors(tmp_path: Path, content: str, code: Code) -> None:
    path = tmp_path / "input.jsonl"
    path.write_text(content)
    with pytest.raises(ContractError) as caught:
        list(read_corpus(path))
    assert caught.value.diagnostic.code == code
    assert caught.value.exit_code == 3


def test_late_gate_and_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    path.write_text(canonical_json(header().model_dump()) + '\n{"record":"bad"}\nnull\n')
    stream = read_corpus(path)
    assert isinstance(next(stream), CorpusHeader)
    assert isinstance(next(stream), CorpusFinding)
    with pytest.raises(ContractError, match="object"):
        next(stream)
    with pytest.raises(ContractError) as caught:
        list(read_corpus(tmp_path / "absent"))
    assert caught.value.diagnostic.code == Code.UNREADABLE


def test_writer_canonical_and_protected(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    records: list[CorpusRecord] = [header(), CorpusMetadata(corpus_id="test-corpus"), CorpusWork(work="GEN", order=1)]
    write_corpus(path, records, [])
    assert path.read_text() == "".join(canonical_json(r.model_dump()) + "\n" for r in records)
    assert record_digest(records[0]) == digest(records[0].model_dump())
    assert record_digest(records[0]).startswith("sha256:")
    before = path.read_bytes()
    with pytest.raises(ContractError) as caught:
        write_corpus(path, records, [path])
    assert caught.value.diagnostic.code == Code.ALIAS
    with writer_lock(path), pytest.raises(ContractError) as caught:
        write_corpus(path, records, [])
    assert caught.value.diagnostic.code == Code.BUSY
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="header"):
        write_corpus(path, records[1:], [])
    assert path.read_bytes() == before


@pytest.mark.parametrize("name", ["usfx-bcv", "ordinal", "wsl"])
def test_fixture_roundtrip(name: str, tmp_path: Path) -> None:
    original = FIXTURES / "valid" / f"{name}.jsonl"
    records = list(read_corpus(original))
    assert not any(isinstance(record, CorpusFinding) for record in records)
    target = tmp_path / "roundtrip.jsonl"
    write_corpus(target, (r for r in records if not isinstance(r, CorpusFinding)), [])
    assert target.read_bytes() == original.read_bytes()
    passages = [r for r in records if isinstance(r, CorpusPassage)]
    assert [r.order for r in passages] == list(range(1, len(passages) + 1))


def test_usfx_literal_wire_records() -> None:
    lines = (FIXTURES / "valid/usfx-bcv.jsonl").read_text().splitlines()
    assert lines[0] == (
        '{"adapter":"usfx/1","corpus_id":"test-corpus","edition":null,"format":"latinitas-corpus",'
        '"language":"la","license":null,"norm_rules":"latin-fold/1","publisher":null,"record":"header",'
        '"source":{"bytes":83,"file_name":"source.txt",'
        '"sha256":"sha256:8bb289ce50920f2eea86aa6e74d01d690e617a9fce9ca15481b574423f22f862"},'
        '"text_canon":1,"url":null,"version":1}'
    )
    assert lines[4] == (
        '{"citation":{"display":"GEN 1:1","parts":[{"unit":"book","value":"GEN"},'
        '{"unit":"chapter","value":"1"},{"unit":"verse","value":"1"}],"scheme":"usfx-bcv"},'
        '"headings":null,"native_ref":"GEN.1.1","norm":"in principio creavit deus caelum et terram.",'
        '"notes":null,"order":1,"passage_hash":'
        '"sha256:f231be94df2e0e4e7899e12485f82668689fe2d6596d0ab5af2b31e484619e73",'
        '"passage_id":"test-corpus:GEN.1.1","record":"passage","source_extra":null,"source_markup":null,'
        '"text":"In principio creavit Deus cælum et terram.","translations":null,"work":"GEN"}'
    )


def test_writer_rejects_float_without_replacing(tmp_path: Path) -> None:
    records = list(read_corpus(FIXTURES / "valid/ordinal.jsonl"))
    passage = next(r for r in records if isinstance(r, CorpusPassage))
    passage.source_extra = {"nested": [1.25]}
    target = tmp_path / "out.jsonl"
    target.write_bytes(b"previous")
    with pytest.raises(ValueError, match="floats"):
        write_corpus(target, (r for r in records if not isinstance(r, CorpusFinding)), [])
    assert target.read_bytes() == b"previous"
    with pytest.raises(ValueError, match="floats"):
        record_digest(passage)


def test_version_zero_and_header_float_are_findings(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps({**header().model_dump(), "version": 0}) + "\n")
    assert isinstance(next(read_corpus(path)), CorpusFinding)
    path.write_text(json.dumps({**header().model_dump(), "source": {"bytes": 1.0}}) + "\n")
    finding = next(read_corpus(path))
    assert isinstance(finding, CorpusFinding) and finding.code == "float_value"


@pytest.mark.parametrize(
    "target,kind,force",
    [
        ("invalid", "parent", True),
        ("schemas", "parent", True),
        ("invalid", "file", True),
        ("schemas", "file", True),
        ("valid", "dangling", False),
        ("valid", "parent", True),
    ],
)
def test_regenerator_refuses_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str, kind: str, force: bool
) -> None:
    script = Path(__file__).parents[2] / "scripts/regen-corpus-fixtures.py"
    main = runpy.run_path(str(script))["main"]
    root = tmp_path / "repo"
    root.mkdir()
    protected = root / "tests/fixtures/goldens/legacy-vulgate"
    protected.mkdir(parents=True)
    victim = protected / "sentinel.json"
    victim.write_bytes(b"unchanged")
    directory = root / ("docs/schemas" if target == "schemas" else f"tests/fixtures/corpus/{target}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    if kind == "parent":
        directory.symlink_to(protected, target_is_directory=True)
    else:
        directory.mkdir()
        name = {"valid": "usfx-bcv.jsonl", "invalid": "non-nfc.jsonl", "schemas": "latinitas-corpus.v1.schema.json"}[
            target
        ]
        (directory / name).symlink_to(victim if kind == "file" else protected / "absent")
    monkeypatch.setitem(main.__globals__, "ROOT", root)
    monkeypatch.setitem(main.__globals__, "FIXTURES", root / "tests/fixtures/corpus")
    monkeypatch.setattr(sys, "argv", [str(script), "--target", target, *(["--force"] if force else [])])
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 2
    assert victim.read_bytes() == b"unchanged"
    assert [p.name for p in protected.iterdir()] == ["sentinel.json"]


@pytest.mark.parametrize("target", ["invalid", "schemas"])
def test_regenerator_force_does_not_modify_hardlink_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    script = Path(__file__).parents[2] / "scripts/regen-corpus-fixtures.py"
    main = runpy.run_path(str(script))["main"]
    root = tmp_path / "repo"
    protected = root / "tests/fixtures/goldens/claims-v1/sentinel.json"
    protected.parent.mkdir(parents=True)
    protected.write_bytes(b"unchanged")
    directory = root / ("docs/schemas" if target == "schemas" else "tests/fixtures/corpus/invalid")
    directory.mkdir(parents=True)
    output = directory / ("non-nfc.jsonl" if target == "invalid" else "latinitas-corpus.v1.schema.json")
    output.hardlink_to(protected)
    monkeypatch.setitem(main.__globals__, "ROOT", root)
    monkeypatch.setitem(main.__globals__, "FIXTURES", root / "tests/fixtures/corpus")
    monkeypatch.setattr(sys, "argv", [str(script), "--target", target, "--force"])
    main()
    assert protected.read_bytes() == b"unchanged"
    assert output.read_bytes() != b"unchanged"
    assert not output.samefile(protected)
