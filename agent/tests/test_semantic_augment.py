"""Synthetic source contracts and mocked HTTP only; no external API calls."""
from copy import deepcopy
import io
import json
from urllib import error

import pytest

from semantic.augment import (CHECKS, CallFailure, MODEL, annotation_prompt, build_source,
    call_teacher, compile_rows, parse_verification, process_source, read_key,
    review_manifest, split_annotation, validate_annotation, verification_prompt)
from semantic.data import validate_row


def source():
    return build_source({"id": "urn:invented:alpha", "code": "fiction", "title": "Alpha",
                         "definition": "Alpha is associated with beta.", "longDefinition": "",
                         "diagnosticCriteria": ""}, {"eligible_train_sources": 148})


def annotation():
    return ("[ENTITIES]\nDisease\tAlpha\tAlpha\nSymptom\tbeta\tbeta\n"
            "[RELATIONS]\nSymptom\tbeta\tis_associated_symptom_of\tDisease\tAlpha\tAlpha is associated with beta.\n")


def verifier(fail=None):
    return "".join(f"CHECK\t{name}\t{'FAIL' if name == fail else 'PASS'}\n" for name in CHECKS)


def verifier_v2(fail=None):
    manifest = review_manifest(source(), validate_annotation(source(), annotation()))
    rows = verifier(fail)
    for kind, key in (("ENTITY", "entities"), ("EDGE", "edges"), ("COVERAGE", "units")):
        rows += "".join(f"{kind}\t{ref}\tPASS\n" for ref in manifest[key])
    return rows


def receipt():
    return {"model": MODEL, "finish_reason": "stop", "usage": {"total_tokens": 10}}


def test_source_only_fields_and_exact_offset_provenance():
    raw = {"id": "urn:x", "title": "Alpha", "definition": "Beta.", "assistant": "forbidden annotation"}
    item = build_source(raw, {})
    assert "forbidden" not in annotation_prompt(item)
    for span in item["source_field_spans"]:
        assert item["source_text"][span["start"]:span["end"]] == item["source_fields"][span["field"]]
    assert item["source_context"].endswith("no_assistant_annotations")


def test_valid_annotation_compiles_both_literal_training_stages():
    item = source()
    parsed = validate_annotation(item, annotation())
    rows = compile_rows(item, parsed, receipt(), {"checks": parse_verification(verifier())})
    assert {row["stage"] for row in rows} == {"entities", "relations"}
    for row in rows:
        validate_row(json.loads(json.dumps(row)))
        assert row["split"] == "train" and not row["metadata"]["human_adjudicated"]
        assert row["metadata"]["unannotated_pair_policy"] == "UNKNOWN_NOT_EXPLICITLY_SUPERVISED"
        assert not row["completion"].startswith(("{", "["))
        assert row["metadata"]["augmentation"]["agreement_is_not_human_truth"]


@pytest.mark.parametrize("replacement", [
    ("Symptom\tbeta\tbeta", "Unknown\tbeta\tbeta"),
    ("Symptom\tbeta\tbeta", "Symptom\tinvented\tinvented"),
    ("is_associated_symptom_of", "invented_relation"),
    ("\tDisease\tAlpha\tAlpha is", "\tSymptom\tbeta\tAlpha is"),
    ("\tAlpha is associated with beta.\n", "\tunquoted evidence\n"),
])
def test_invalid_rows_are_rejected_whole_source(replacement):
    with pytest.raises(ValueError):
        validate_annotation(source(), annotation().replace(*replacement))


def test_title_alone_is_not_relation_evidence():
    with pytest.raises(ValueError, match="body-field"):
        validate_annotation(source(), annotation().replace("\tAlpha is associated with beta.\n", "\tdefinition:\n"))


@pytest.mark.parametrize("bad", ["", "```\n" + annotation(), annotation() + "[RELATIONS]\n",
                                  annotation().replace("[RELATIONS]\n", "[RELATIONS]extra\n")])
def test_sections_are_strict_and_never_repaired(bad):
    with pytest.raises(ValueError):
        split_annotation(bad)


def test_explicit_container_grammar_allows_only_one_boundary_blank():
    original = annotation()
    one = original.replace("\n[RELATIONS]", "\n\n[RELATIONS]")
    with pytest.raises(ValueError):
        validate_annotation(source(), one)
    assert split_annotation(one, allow_boundary_blank=True) == split_annotation(original)
    validate_annotation(source(), one, allow_boundary_blank=True)
    two = original.replace("\n[RELATIONS]", "\n\n\n[RELATIONS]")
    with pytest.raises(ValueError):
        validate_annotation(source(), two, allow_boundary_blank=True)
    internal = original.replace("Disease\tAlpha\tAlpha\n", "Disease\tAlpha\tAlpha\n\n")
    with pytest.raises(ValueError):
        validate_annotation(source(), internal, allow_boundary_blank=True)


def test_empty_relations_make_no_negative_training_row():
    parsed = validate_annotation(source(), annotation().split("[RELATIONS]")[0] + "[RELATIONS]\n")
    rows = compile_rows(source(), parsed, {}, {})
    assert [row["stage"] for row in rows] == ["entities"]


@pytest.mark.parametrize("text", [verifier() + "\n", verifier().replace("PASS", "yes", 1), verifier().replace("source_support", "other", 1), "{}"])
def test_verifier_contract_fails_closed(text):
    with pytest.raises(ValueError):
        parse_verification(text)


def test_verifier_failure_does_not_publish_training_rows():
    responses = iter([(annotation(), receipt()), (verifier_v2("major_relations_coverage"), receipt())])
    result = process_source(source(), "secret", caller=lambda prompt, key: next(responses))
    assert result["status"] == "verifier_rejected" and "rows" not in result
    assert len(result["api_calls"]) == 2


def test_malformed_verifier_kept_privately_for_failure_audit():
    responses = iter([(annotation(), receipt()), ("invalid format", receipt())])
    result = process_source(source(), "SECRET", caller=lambda prompt, key: next(responses))
    assert result["status"] == "validation_rejected" and result["phase"] == "verification_validation"
    assert result["verification"] == "invalid format"
    assert len(result["api_calls"]) == 2 and "SECRET" not in json.dumps(result)


def test_two_fresh_calls_accept_source_and_no_key_in_artifacts():
    prompts = []
    responses = iter([(annotation(), receipt()), (verifier_v2(), receipt())])
    def caller(prompt, key):
        assert key == "SECRET-never-log"
        prompts.append(prompt)
        return next(responses)
    result = process_source(source(), "SECRET-never-log", caller=caller)
    assert result["status"] == "accepted_silver" and len(prompts) == 2
    assert "PROPOSED ANNOTATION" not in prompts[0] and "PROPOSED ANNOTATION" in prompts[1]
    assert "SECRET-never-log" not in json.dumps(result)


def test_review_manifest_requires_each_entity_edge_and_input_unit():
    manifest = review_manifest(source(), validate_annotation(source(), annotation()))
    decisions = parse_verification(verifier_v2(), manifest)
    assert len(decisions) == len(CHECKS) + 2 + 1 + 1
    with pytest.raises(ValueError, match="omitted"):
        parse_verification(verifier(), manifest)
    with pytest.raises(ValueError):
        parse_verification(verifier_v2() + next(line for line in verifier_v2().splitlines() if line.startswith("EDGE")) + "\n", manifest)


def test_failed_paragraph_requires_literal_missing_quote():
    manifest = review_manifest(source(), validate_annotation(source(), annotation()))
    unit = next(iter(manifest["units"]))
    response = verifier_v2().replace(f"COVERAGE\t{unit}\tPASS", f"COVERAGE\t{unit}\tFAIL")
    with pytest.raises(ValueError, match="disagree"):
        parse_verification(response, manifest)
    result = parse_verification(response + f"MISSING\t{unit}\tbeta\tmajor symptom omitted\n", manifest)
    assert result[f"COVERAGE:{unit}"] == "FAIL"
    with pytest.raises(ValueError, match="actual input"):
        parse_verification(response + f"MISSING\t{unit}\tinvented quote\tmissing\n", manifest)


def test_v2_explicitly_addresses_observed_semantic_gate_failures():
    for prompt in (annotation_prompt(source()), verification_prompt(source(), annotation())):
        assert "recommended_examination" in prompt and "list-introduction" in prompt
        assert "duration" in prompt and "negation" in prompt
    row = compile_rows(source(), validate_annotation(source(), annotation()), {}, {})[0]
    assert row["metadata"]["augmentation"]["training_admission"] == "pilot_pending_independent_review"


class Response:
    def __init__(self, finish="stop"):
        self.body = json.dumps({"id": "response", "model": MODEL, "usage": {"total_tokens": 3},
            "choices": [{"finish_reason": finish, "message": {"content": "result"}}]}).encode()
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self):
        return self.body


def test_http_payload_and_only_transport_retries():
    seen = []
    def opener(req, timeout):
        seen.append(req)
        if len(seen) < 3:
            raise error.URLError("SECRET-server-error")
        return Response()
    content, info = call_teacher("prompt", "SECRET", opener=opener, sleeper=lambda _: None)
    assert content == "result" and info["transport_attempts"] == len(seen) == 3
    payload = json.loads(seen[-1].data)
    assert payload["thinking"] == {"type": "disabled"} and payload["temperature"] == .2
    assert payload["model"] == MODEL and payload["max_tokens"] == 8192
    assert payload["messages"] == [{"role": "user", "content": "prompt"}]
    assert "SECRET" not in json.dumps(info)


def test_transport_error_is_bounded_and_redacted():
    count = 0
    def opener(*args, **kwargs):
        nonlocal count
        count += 1
        raise error.URLError("SECRET-server-error")
    with pytest.raises(CallFailure) as exc:
        call_teacher("prompt", "SECRET", opener=opener, sleeper=lambda _: None)
    assert count == 3 and str(exc.value) == "transport_error"


def test_unauthorized_is_not_retried_or_exposed():
    seen = []
    def opener(req, timeout):
        seen.append(1)
        raise error.HTTPError("https://example", 401, "SECRET", {}, io.BytesIO(b"SECRET"))
    with pytest.raises(CallFailure) as exc:
        call_teacher("prompt", "SECRET", opener=opener, sleeper=lambda _: None)
    assert seen == [1] and exc.value.status == 401 and str(exc.value) == "http_error"


@pytest.mark.parametrize("finish", ["length", "content_filter", None])
def test_non_stop_rejected_with_usage_receipt(finish):
    with pytest.raises(CallFailure) as exc:
        call_teacher("prompt", "SECRET", opener=lambda *a, **k: Response(finish))
    assert exc.value.code == "non_stop_completion" and exc.value.receipt["finish_reason"] == finish


def test_key_must_be_private(tmp_path):
    key = tmp_path / "key"
    key.write_text("SECRET\n")
    key.chmod(0o600)
    assert read_key(key) == "SECRET"
    key.chmod(0o644)
    with pytest.raises(ValueError):
        read_key(key)
