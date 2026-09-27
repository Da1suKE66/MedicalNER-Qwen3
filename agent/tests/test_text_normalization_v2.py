"""Synthetic-only normalization/offset tests; no frozen medical corpus needed."""

import copy
import json
from pathlib import Path
import random
import unicodedata

import pytest

import data_repair.text_normalization as normalization_module
from data_repair.text_normalization import (
    find_mentions,
    map_range,
    normalize_with_offsets,
    repair_graph_text,
)


def test_stable_json_api_identity_and_empty_input():
    for text in ("", "No pain; 1,000 mg, 2.5 kg/m2.\n", " null NaN NULL nullify "):
        mapping = normalize_with_offsets(text)
        assert set(mapping) == {"text", "original_text", "offsets", "changes", "offset_unit"}
        assert mapping["text"] == mapping["original_text"] == text
        assert mapping["offset_unit"] == "unicode_codepoint"
        assert mapping["offsets"] == [[i, i + 1] for i in range(len(text))]
        assert mapping["changes"] == []
        round_trip = json.loads(json.dumps(mapping))
        assert map_range(round_trip, 0, len(text)) == (0, len(text))


@pytest.mark.parametrize(("raw", "expected"), [
    ("e\u0301", "é"),
    ("A\u030a\u0301", "Ǻ"),
    ("\u212b", "Å"),
    ("a\u0315\u0300", "à\u0315"),
    ("\u0315\u0300", "\u0300\u0315"),
    ("\u0344", "\u0308\u0301"),
    ("\u1100\u1161\u11a8", "각"),
    ("가\u11a8", "각"),
    ("\uff76\uff9e", "ガ"),
    ("\u09c7\u09be", "\u09cb"),
    ("ＡＢＣ１２３", "ABC123"),
    ("kg/m²", "kg/m2"),
    ("\ufb03", "ffi"),
    ("㎏", "kg"),
    ("Ⅳ ①", "IV 1"),
    ("A\r\nB\rC\n", "A\nB\nC\n"),
    ("x\u00a0y\tz", "x y\tz"),
])
def test_nfkc_and_line_endings_match_unicode_reference(raw, expected):
    mapping = normalize_with_offsets(raw, repair=False)
    assert mapping["text"] == expected
    assert expected == unicodedata.normalize("NFKC", raw).replace("\r\n", "\n").replace("\r", "\n")
    assert mapping["original_text"] == raw
    assert len(mapping["offsets"]) == len(expected)
    assert map_range(mapping, 0, len(expected)) == (0, len(raw))
    for index, pair in enumerate(mapping["offsets"]):
        assert map_range(mapping, index, index + 1) == tuple(pair)
    again = normalize_with_offsets(expected, repair=False)
    assert again["text"] == expected and again["changes"] == []


def test_multiple_length_changes_map_back_to_exact_original_mentions():
    raw = "Ａ e\u0301 \ufb03 ㎏/m² Pregnullcy\r\nwithout pain."
    mapping = normalize_with_offsets(raw)
    assert mapping["text"] == "A é ffi kg/m2 Pregnancy\nwithout pain."
    cases = [
        ("A", "Ａ"), ("É", "e\u0301"), ("FFI", "\ufb03"), ("kg/m²", "㎏/m²"),
        ("pregnancy", "Pregnullcy"), ("without \t pain", "without pain"),
        ("pregnancy without", "Pregnullcy\r\nwithout"),
    ]
    for needle, expected_raw in cases:
        hits = find_mentions(needle, mapping["text"])
        assert len(hits) == 1
        a, b = map_range(mapping, *hits[0])
        assert raw[a:b] == expected_raw


def test_atomic_expansion_composition_and_repair_boundaries():
    assert normalize_with_offsets("\ufb03")["offsets"] == [[0, 1]] * 3
    assert normalize_with_offsets("e\u0301")["offsets"] == [[0, 2]]
    mapping = normalize_with_offsets("pregnullcy")
    assert mapping["offsets"] == [[0, len("pregnullcy")]] * len("pregnancy")


def test_crlf_range_and_normalization_does_not_flatten_document_structure():
    raw = "\tTitle\r\n\r\n  A\t B  \rEnd"
    mapping = normalize_with_offsets(raw)
    assert mapping["text"] == "\tTitle\n\n  A\t B  \nEnd"
    first = mapping["text"].index("\n")
    assert raw[slice(*map_range(mapping, first, first + 1))] == "\r\n"
    a, b = find_mentions("A B", mapping["text"])[0]
    assert raw[slice(*map_range(mapping, a, b))] == "A\t B"


def test_map_range_covers_canonical_reordering_not_just_endpoint_offsets():
    mapping = normalize_with_offsets("a\u0315\u0300")
    assert mapping["text"] == "à\u0315"
    assert mapping["offsets"] == [[0, 3], [1, 2]]
    assert map_range(mapping, 0, 2) == (0, 3)
    assert map_range(mapping, 0, 1) == (0, 3)
    assert map_range(mapping, 1, 2) == (1, 2)


def test_change_grouping_looks_past_adjacent_unit_for_reordered_marks():
    raw = "a\u0344\u0327"
    mapping = normalize_with_offsets(raw, repair=False)
    assert mapping["text"] == unicodedata.normalize("NFKC", raw)
    assert len(mapping["changes"]) == 1
    change = mapping["changes"][0]
    assert change["original_start"] == 0 and change["original_end"] == len(raw)
    assert change["before"] == raw and change["after"] == mapping["text"]


def test_offsets_are_codepoints_not_bytes_or_utf16_units():
    raw = "\U0001f48aＡ"
    mapping = normalize_with_offsets(raw)
    assert len(mapping["text"]) == 2
    assert mapping["offsets"] == [[0, 1], [1, 2]]
    assert raw[slice(*map_range(mapping, *find_mentions("a", mapping["text"])[0]))] == "Ａ"


def test_changes_index_final_and_original_views_and_replay_as_edits():
    raw = "xＡ e\u0301 \ufb03 pregnullcy\r\nanullkastic!"
    mapping = normalize_with_offsets(raw)
    reconstructed = raw
    for event in reversed(mapping["changes"]):
        a, b = event["original_start"], event["original_end"]
        assert raw[a:b] == event["before"]
        assert mapping["text"][event["start"]:event["end"]] == event["after"]
        assert event["kind"] == "normalization" and event["applied"]
        reconstructed = reconstructed[:a] + event["after"] + reconstructed[b:]
    assert reconstructed == mapping["text"]
    assert normalize_with_offsets(mapping["text"])["changes"] == []


def test_repair_switch_and_original_mapping_basis_are_explicit():
    raw = "Ｐｒｅｇｎｕｌｌｃｙ\r\n"
    uncorrected = normalize_with_offsets(raw, repair=False)
    corrected = normalize_with_offsets(raw)
    assert uncorrected["text"] == "Pregnullcy\n"
    assert corrected["text"] == "Pregnancy\n"
    assert map_range(corrected, 0, len("Pregnancy")) == (0, len("Ｐｒｅｇｎｕｌｌｃｙ"))
    second = normalize_with_offsets(corrected["text"])
    assert second["original_text"] != raw  # Never discard the first mapping.
    assert second["text"] == corrected["text"] and second["changes"] == []


def test_all_audited_dictionary_entries_and_case_variants_are_repaired():
    path = Path(__file__).resolve().parents[2] / "schemas/v2.0.0/schema.json"
    pairs = json.loads(path.read_text())["text_corrections"]
    assert len(pairs) == 22
    for before, after in pairs.items():
        for variant, expected in [(before, after), (before.upper(), after.upper()),
                                  (before.capitalize(), after.capitalize())]:
            mapping = normalize_with_offsets(variant)
            assert mapping["text"] == expected
            assert map_range(mapping, 0, len(expected)) == (0, len(variant))


@pytest.mark.parametrize("text", [
    "apregnullcy", "pregnullcyx", "_pregnullcy", "pregnullcy_",
    "1pregnullcy", "pregnullcy2", "pregnullcy\u0338", "\u0338pregnullcy",
    "null nullify nullable annulled NaN 1,000 mg without symptoms",
])
def test_no_subword_global_null_or_negation_numeric_rewrites(text):
    assert normalize_with_offsets(text)["text"] == text
    assert normalize_with_offsets(text)["changes"] == []


@pytest.mark.parametrize(("needle", "text", "slices"), [
    ("kg/m²", "unit: ＫＧ/ｍ2.", ["ＫＧ/ｍ2"]),
    ("ＫＧ/ｍ2", "unit: kg/m².", ["kg/m²"]),
    ("café", "Cafe\u0301, CAFÉ", ["Cafe\u0301", "CAFÉ"]),
    ("Straße", "STRASSE; Straße", ["STRASSE", "Straße"]),
    ("A B", "a\r\n\tb / A\u00a0 B", ["a\r\n\tb", "A\u00a0 B"]),
    (" \tA B\n", "a b", ["a b"]),
    ("(pain)", "x(pain)y", ["(pain)"]),
    ("[A]", "x[A]y", ["[A]"]),
    ("C++", "C++/C++ xC++", ["C++", "C++"]),
    (".*", "x.*y", [".*"]),
    ("pain", "painting pain pains _pain pain2", ["pain"]),
    ("a a", "a a a", ["a a", "a a"]),
    ("SS", "ß", ["ß"]),
    ("Ⅳ", "IV; Ⅳ", ["IV", "Ⅳ"]),
    ("", "pain", []),
    (" \t\r\n", "pain", []),
])
def test_symmetric_nfkc_case_whitespace_and_conditional_boundaries(needle, text, slices):
    hits = find_mentions(needle, text)
    assert [text[a:b] for a, b in hits] == slices
    assert hits == sorted(set(hits))


@pytest.mark.parametrize(("needle", "text"), [
    ("1000 mg", "1,000 mg"),
    ("1.5 mg", "15 mg"),
    ("2 mg", "12 mg"),
    ("with pain", "without pain"),
    ("without pain", "with pain"),
    ("Disorder Alpha", "Disorder-Alpha"),
    ("Bipolar I", "Bipolar II"),
    ("e", "é"),
    ("pain", "pain\u0338"),
    ("s", "ß"),
    ("(a)b", "x(a)bc"),
    ("a(b)", "xa(b)y"),
    ("pregnancy", "pregnullcy"),  # Finder never silently overrides repair=False.
])
def test_matching_does_not_invent_aliases_or_match_inside_words(needle, text):
    assert find_mentions(needle, text) == []


def test_descriptive_graph_fields_definition_and_nested_lists_are_repaired():
    graph = {
        "entities": [{
            "id": "pregnullcy", "label": "Disease", "name": "Pregnullcy",
            "properties": {
                "Definition": ["no pregnullcy", ["Ｍａｉｎｔｅｎｕｌｌｃｅ", None],
                               {"clauses": ["finullcial"], "id": "pregnullcy"}],
                "diagnosticCriteria": "without pregnullcy; 1,000 mg",
                "Description": {"sentences": ["anullkastic"]},
            },
        }],
        "relations": [{
            "source": "pregnullcy", "target": "anullkastic",
            "relation": "predominullt", "evidence_sentence": ["Pregnullcy\r\n", None],
        }],
    }
    before = copy.deepcopy(graph)
    fixed, changes = repair_graph_text(graph)
    assert graph == before and fixed is not graph
    node = fixed["entities"][0]
    assert node["name"] == "Pregnancy"
    assert node["properties"]["Definition"] == [
        "no pregnancy", ["Maintenance", None], {"clauses": ["financial"], "id": "pregnullcy"},
    ]
    assert node["properties"]["diagnosticCriteria"] == "without pregnancy; 1,000 mg"
    assert node["properties"]["Description"] == {"sentences": ["anankastic"]}
    assert fixed["relations"][0]["evidence_sentence"] == ["Pregnancy\n", None]
    assert node["id"] == "pregnullcy"
    assert fixed["relations"][0]["relation"] == "predominullt"
    assert all(c["applied"] for c in changes)
    assert any(c["path"] == "/entities/0/properties/Definition/0" for c in changes)
    again, second_changes = repair_graph_text(fixed)
    assert again == fixed and second_changes == []
    fixed["entities"].append({})
    assert graph == before


@pytest.mark.parametrize("key", [
    "id", "ID", "patientId", "entity_id", "code", "ICDCode", "ICD_11_Code",
    "uri", "URL", "foundationReference", "relation", "relation_type", "label",
    "source", "target", "source_id", "$schema", "schema_version", "ontology",
    "original_text", "original_record", "original_spans", "provenance",
    "original_source", "original_payload", "raw_text", "raw_source",
    "entity_type", "entityLabel", "relation_schema",
])
def test_protected_graph_subtrees_win_even_inside_description(key):
    value = {"Description": {key: {"Definition": "pregnullcy", "name": "Ａ"}}}
    fixed, changes = repair_graph_text(value)
    assert fixed == value and changes == []
    assert fixed is not value and fixed["Description"] is not value["Description"]


def test_nulls_numbers_negation_and_labels_are_not_retyped():
    graph = {
        "name": None,
        "Definition": [None, False, 12, 1.5, "null NaN nullify", "without pain 1,000 mg"],
        "code": "001",
        "type": "Ｐｒｅｇｎｕｌｌｃｙ",
    }
    fixed, changes = repair_graph_text(graph)
    assert fixed == graph and changes == []
    assert json.loads(json.dumps(fixed))["name"] is None


def test_unknown_fields_are_diagnosed_not_repaired_and_paths_are_escaped():
    graph = {"properties": {"custom/~value": ["pregnullcy", None], "plain": "safe"}}
    fixed, changes = repair_graph_text(graph)
    assert fixed == graph
    assert len(changes) == 1
    change = changes[0]
    assert change["path"] == "/properties/custom~1~0value/0"
    assert change["kind"] == "unclassified_text_field"
    assert change["applied"] is False
    assert change["before"] == "pregnullcy" and change["after"] == "pregnancy"
    again, diagnostics = repair_graph_text(fixed)
    assert again == graph and diagnostics == changes


@pytest.mark.parametrize(("start", "end"), [(-1, 1), (2, 1), (0, 4)])
def test_invalid_ranges_raise_without_clipping(start, end):
    with pytest.raises(ValueError):
        map_range(normalize_with_offsets("abc"), start, end)


def test_empty_ranges_have_explicit_ambiguity_policy():
    mapping = normalize_with_offsets("a\ufb03b")
    assert map_range(mapping, 0, 0) == (0, 0)
    assert map_range(mapping, 1, 1) == (1, 1)
    assert map_range(mapping, 2, 2) == (1, 2)
    assert map_range(mapping, 4, 4) == (2, 2)
    assert map_range(mapping, 5, 5) == (3, 3)
    assert map_range(normalize_with_offsets(""), 0, 0) == (0, 0)


def test_invalid_types_and_malformed_mappings_raise():
    for raw in (None, 1, [], {}):
        with pytest.raises(TypeError):
            normalize_with_offsets(raw)
    with pytest.raises(TypeError):
        normalize_with_offsets("x", repair=1)
    with pytest.raises(TypeError):
        find_mentions(None, "text")
    with pytest.raises(TypeError):
        find_mentions("text", None)
    with pytest.raises(TypeError):
        map_range(normalize_with_offsets("a"), True, 1)
    with pytest.raises(ValueError):
        map_range({"text": "a", "original_text": "a", "offsets": []}, 0, 1)
    with pytest.raises(ValueError):
        map_range({"text": "a", "original_text": "a", "offsets": [[0, 5]]}, 0, 1)


def test_repeated_document_queries_reuse_one_bounded_comparison_view(monkeypatch):
    comparison = normalization_module._comparison_units
    comparison.cache_clear()
    calls = {}
    original = normalization_module._nfkc_units

    def counted(text):
        calls[text] = calls.get(text, 0) + 1
        return original(text)

    monkeypatch.setattr(normalization_module, "_nfkc_units", counted)
    document = "Alpha beta gamma.\r\nＡＬＰＨＡ"
    try:
        for _ in range(4):
            for needle in ("alpha", "beta", "gamma"):
                assert find_mentions(needle, document)
        assert calls[document] == 1
        assert sum(calls.values()) == 4  # Three queries and one document.
        for index in range(32):
            comparison(f"distinct synthetic document {index}")
        info = comparison.cache_info()
        assert info.maxsize == info.currsize == 8
    finally:
        comparison.cache_clear()


def test_cached_comparison_is_immutable_and_hits_are_caller_owned():
    comparison = normalization_module._comparison_units
    comparison.cache_clear()
    try:
        view = comparison("pain pain")
        assert isinstance(view, tuple)
        with pytest.raises(TypeError):
            view[0] = view[0]
        with pytest.raises(AttributeError):
            view[0].start = 99
        hits = find_mentions("pain", "pain pain")
        hits.clear()
        assert find_mentions("pain", "pain pain") == [(0, 4), (5, 9)]
    finally:
        comparison.cache_clear()


def test_deterministic_unicode_composition_fuzz_and_edit_replay():
    rng = random.Random(20260915)
    alphabet = list("abc 12\t\r\n") + [
        "\u0301", "\u0300", "\u0315", "\u0327", "\u0345", "\u0344", "\u09c7",
        "\u09be", "\u1100", "\u1161", "\u11a8", "\ufb03", "\uff76", "\uff9e",
        "\u212b", "㎏", "²", "Ａ", "é", "ß", "\u00a0",
    ]
    for _ in range(300):
        raw = "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 40)))
        mapping = normalize_with_offsets(raw, repair=False)
        expected = unicodedata.normalize("NFKC", raw).replace("\r\n", "\n").replace("\r", "\n")
        assert mapping["text"] == expected
        assert map_range(mapping, 0, len(expected)) == (0, len(raw))
        replayed = raw
        for change in reversed(mapping["changes"]):
            a, b = change["original_start"], change["original_end"]
            replayed = replayed[:a] + change["after"] + replayed[b:]
        assert replayed == expected
        assert normalize_with_offsets(expected, repair=False)["changes"] == []
