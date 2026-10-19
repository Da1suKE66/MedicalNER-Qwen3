"""Synthetic source-provenance tests; no private medical source data is stored."""

from dataclasses import asdict
import json

import pytest

from data_repair.source_provenance import source_view
from kg_agent.source import SourceRouter


def assert_fragment(view, fragment):
    assert view["raw_user"][fragment["start"]:fragment["end"]] == fragment["raw"]
    assert view["raw_payload"][fragment["payload_start"]:fragment["payload_end"]] == fragment["raw"]
    assert fragment["start"] == view["payload_start"] + fragment["payload_start"]


def icd_text(items, *, pretty=False):
    return (
        "http://id.who.int/icd/release/11/2025-01/mms/123/unspecified\n"
        "6B20.Z\nCondition Alpha, unspecified\ncategory\n"
        '["http://id.who.int/icd/release/11/2025-01/mms/123"]\n'
        + json.dumps(items, ensure_ascii=False, indent=2 if pretty else None)
    )


def test_full_user_offsets_are_not_nfkc_or_crlf_offsets():
    prefix = "Instruction 🧭: return the source.\r\n\r\n  Medical text:\t\r\n"
    payload = "\r\n  Ｆatigue and ﬁndings: ①.\r\nKeep trailing space.  \r\n"
    user = prefix + payload
    view = source_view(user)
    assert view["raw_user"] == user
    assert view["payload_start"] == len(prefix)
    assert view["payload_end"] == len(user)
    assert user[view["payload_start"]:view["payload_end"]] == payload == view["raw_payload"]
    assert "Ｆ" in view["raw_payload"] and "\r\n" in view["raw_payload"]
    assert "Ｆ" not in view["routed"]["raw"] and "\r\n" not in view["routed"]["raw"]
    assert view["routed"] == asdict(SourceRouter().route(user))
    assert not view["routed_is_original"] and not view["routed_semantics_verified"]
    assert view["routed_span_offset_space"] == "routed.raw_only_not_original_user"


@pytest.mark.parametrize("text", ["", "  symptoms can start in childhood.\r\n", "Medical text is not a marker."])
def test_no_wrapper_preserves_entire_input(text):
    view = source_view(text)
    assert view["raw_payload"] == text
    assert (view["payload_start"], view["payload_end"]) == (0, len(text))
    assert view["wrapper"] is None
    assert not view["flags"]


@pytest.mark.parametrize("ending", ["\n", "\r\n", "\r"])
def test_marker_recognition_preserves_each_line_ending(ending):
    prefix = f"Instruction{ending}mEdIcAl TeXt:{ending}"
    view = source_view(prefix + "raw" + ending)
    assert view["payload_start"] == len(prefix)
    assert view["raw_payload"] == "raw" + ending


def test_only_first_standalone_marker_removed():
    user = "Instruction\nMedical text:\nOriginal block.\nMedical text:\nSecond block."
    view = source_view(user)
    assert view["raw_payload"] == "Original block.\nMedical text:\nSecond block."
    flag = next(f for f in view["flags"] if f["code"] == "multiple_payload_markers")
    assert_fragment(view, flag)


def test_json_dict_payload_preserves_unknown_fields_and_exact_nested_item_offsets():
    item = {"label": "Ｆinding", "foundationReference": "https://id.who.int/icd/entity/7", "extra": {"keep": [None, False, 4]}}
    data = {"code": "6A00", "title": "Ａlpha", "unknown": {"lookup/list": [item]}, "unrelated": {"all": "kept"}}
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\r\n "
    view = source_view("Prompt 🌿\nMedical text:\n" + payload)
    assert view["source_type"] == "structured_icd"
    assert view["raw_payload"] == payload
    assert view["json_payload"] == data
    assert view["json_blocks"][0]["value"] == data
    listing = view["labelled_lists"][0]
    assert listing["json_pointer"] == "/unknown/lookup~1list"
    assert listing["declared_field"] == "lookup/list"
    assert listing["items"][0]["value"] == item
    assert_fragment(view, listing)
    assert_fragment(view, listing["items"][0])
    assert "Ｆinding" in listing["items"][0]["raw"]


@pytest.mark.parametrize("pretty", [False, True])
def test_source355_shaped_mixed_reference_list_keeps_all_four_items(pretty):
    ref = "http://id.who.int/icd/entity/123"
    items = [
        {"foundationReference": "", "label": "Condition Alpha, unspecified", "extra": 1},
        {"foundationReference": ref, "label": "Condition Alpha", "unknown": {"a": [1, 2]}},
        {"foundationReference": ref, "label": "anullkastic example", "deprecated": False},
        {"foundationReference": ref, "label": "Alternative name", "keep_null": None},
    ]
    user = "Instruction\nMedical text:\n" + icd_text(items, pretty=pretty)
    view = source_view(user)
    assert view["source_type"] == "structured_icd"
    assert view["routed"] == asdict(SourceRouter().route(user))
    assert "exclusions" not in view["routed"]["metadata"]  # Frozen router behavior.
    assert len(view["json_blocks"]) == 2  # Includes the non-labelled reference list.
    assert len(view["labelled_lists"]) == 1
    listing = view["labelled_lists"][0]
    assert listing["line_index"] == 5 and listing["declared_field"] is None
    assert [x["index"] for x in listing["items"]] == [0, 1, 2, 3]
    assert [x["value"] for x in listing["items"]] == items
    assert [x["has_valid_reference"] for x in listing["items"]] == [False, True, True, True]
    for item in listing["items"]:
        assert_fragment(view, item)
        assert item["semantic_status"] == "review"
        assert "list_semantics_unverified" in item["review_reasons"]
    assert "anullkastic" in view["raw_payload"]  # No word repair in this API.


def test_unnamed_fully_referenced_list_does_not_gain_new_semantics():
    items = [{"foundationReference": "http://id.who.int/icd/entity/123", "label": "Alpha"}]
    view = source_view(icd_text(items))
    # Preserve compatibility even when the OLD router makes its heuristic claim.
    assert "exclusions" in view["routed"]["metadata"]
    listing = view["labelled_lists"][0]
    assert listing["declared_field"] is None
    assert listing["semantic_status"] == "review"
    assert listing["items"][0]["semantic_status"] == "review"
    assert "exclusions" not in view and "relations" not in view and "entities" not in view


def test_explicit_named_field_is_preserved_not_promoted_to_graph_edges():
    data = {"icd_code": "6A00", "exclusions": [{"label": "Alpha", "foundation_reference": "https://example.org/1"}]}
    view = source_view(json.dumps(data))
    listing = view["labelled_lists"][0]
    assert listing["declared_field"] == "exclusions"
    assert listing["items"][0]["references"][0]["field"] == "foundation_reference"
    assert listing["items"][0]["has_valid_reference"]
    assert listing["semantic_status"] == "review" and "relations" not in view


def test_mixed_item_types_duplicate_items_and_all_fields_survive():
    labelled = {"label": "Same", "foundationReference": "https://example.org/ref", "unrecognized": [1, {"x": True}]}
    items = [labelled, labelled, None, "opaque item", 42, {"not_a_label": "keep"}]
    view = source_view(icd_text(items))
    saved = view["labelled_lists"][0]["items"]
    assert [x["value"] for x in saved] == items
    assert [x["index"] for x in saved] == list(range(len(items)))
    assert saved[0]["raw"] == saved[1]["raw"] and saved[0]["start"] != saved[1]["start"]
    assert "non_object_item" in saved[2]["review_reasons"]
    for item in saved:
        assert_fragment(view, item)


@pytest.mark.parametrize("reference", [None, "", "  ", 1, False, "not-a-uri", "https://", "https://host:bad/ref", "http://host/a b", "javascript:alert(1)"])
def test_invalid_reference_is_retained_and_flagged_per_item(reference):
    view = source_view(icd_text([{"label": "Alpha", "foundationReference": reference}]))
    item = view["labelled_lists"][0]["items"][0]
    assert item["value"]["foundationReference"] == reference
    assert not item["has_valid_reference"]
    assert "missing_or_invalid_reference" in item["review_reasons"]


def test_string_escapes_and_duplicate_json_key_bytes_are_preserved_as_raw_text():
    text = '{"icd_code":"6A00","lookup":[{"label":"A\\u0042","x":1,"x":2}]}'
    view = source_view(text)
    item = view["labelled_lists"][0]["items"][0]
    assert item["raw"] == '{"label":"A\\u0042","x":1,"x":2}'
    assert item["value"]["label"] == "AB"
    assert view["raw_payload"] == text
    assert_fragment(view, item)


def test_invalid_json_is_retained_without_inventing_empty_values():
    user = 'Medical text:\n{"icd_code":"6A00", broken}\r\n'
    view = source_view(user)
    assert view["raw_payload"] == '{"icd_code":"6A00", broken}\r\n'
    assert not view["json_blocks"][0]["decoded"]
    assert view["json_blocks"][0]["parse_error"]
    assert view["labelled_lists"] == []
    assert view["flags"][0]["code"] == "unparsed_json_block"
    assert_fragment(view, view["json_blocks"][0])


def test_source47_shaped_suffix_only_emits_uncertain_flag_no_prefix_guess():
    payload = "rment (i.e., reading and writing) is described in this source."
    view = source_view("Instruction\nMedical text:\n" + payload)
    flag = next(f for f in view["flags"] if f["code"] == "suspected_start_truncation")
    assert flag["uncertain"] and flag["raw"] == "rment"
    assert view["issues"] == view["flags"] and flag["needs_review"]
    assert all("code" in issue and "needs_review" in issue for issue in view["issues"])
    assert view["raw_payload"] == payload
    assert "replacement" not in flag
    assert_fragment(view, flag)


@pytest.mark.parametrize("payload", ["symptoms begin in childhood.", "impairment is common.", "reading problems occur.", "the source has no terminal punctuation"])
def test_normal_lowercase_or_unpunctuated_text_is_not_assumed_truncated(payload):
    view = source_view(payload)
    assert not any("truncation" in f["code"] for f in view["flags"])


def test_block_and_boundary_risks_have_original_offsets_without_repairs():
    payload = "## Features:\r\nSymptoms persist.Another sentence follows.\r\n## Features:\r\nexperi-\nences vary.\nThe block ends at impair-"
    view = source_view("Header\nMedical text:\n" + payload)
    codes = {f["code"] for f in view["flags"]}
    assert codes == {"duplicate_header", "no_space_sentence_boundary", "suspected_word_break", "suspected_end_truncation"}
    assert view["raw_payload"] == payload
    for flag in view["flags"]:
        assert flag["uncertain"]
        assert_fragment(view, flag)


def test_abbreviations_decimals_urls_and_chinese_punctuation_are_not_no_space_risks():
    view = source_view("Dr.Smith noted 3.5 mg, e.g.Example. Visit https://host.test/Path.Next or a.Path.Next@example.org. 中文。下一句。")
    assert "no_space_sentence_boundary" not in {f["code"] for f in view["flags"]}


def test_non_string_input_requires_original_user_not_an_already_parsed_dict():
    with pytest.raises(TypeError, match="original user string"):
        source_view({"icd_code": "6A00"})
