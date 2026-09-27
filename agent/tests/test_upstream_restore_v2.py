import copy

from data_repair.release_v2 import canonical, prepare_record
from data_repair.upstream_restore import source_restorations
from kg_agent.contracts import ONTOLOGY


URI = "http://id.who.int/icd/release/11/2025-01/mms/12345"
FULL = "The document describes impairment (for a synthetic example). Feature A accompanies Condition B."
CHUNK = "rment (for a synthetic example). Feature A accompanies Condition B."


def record(user, graph):
    return {"messages": [{"role":"system", "content":ONTOLOGY["version"]},
                         {"role":"user", "content":"Medical text:\n"+user},
                         {"role":"assistant", "content":canonical(graph)}]}


def inputs():
    nodes = [{"id":"S", "name":"Feature A", "label":"Symptom"},
             {"id":"D", "name":"Condition B", "label":"Disease"}]
    graph = {"entities":nodes,"relations":[{"source":"S","target":"D","relation":"is_associated_symptom_of",
                                           "evidence":"Feature A accompanies Condition B."}]}
    records = [record(canonical({"uri":URI, "code":"XX01", "title":"Condition B"}),{"entities":[nodes[1]],"relations":[]}),
               record(CHUNK,graph)]
    upstream = {"entities":[{"id":URI,"code":"XX01","diagnosticCriteria":FULL}]}
    return records,upstream


def test_unique_full_chunk_same_release_identity_recovers_original_prefix():
    records, upstream = inputs()
    before = copy.deepcopy(records)
    restored = source_restorations(records, upstream, [0,0])
    assert set(restored) == {1}
    assert restored[1]["effective_user"] == "Medical text:\n"+FULL
    item = prepare_record(records[1],1,0,"train",restoration=restored[1])
    assert item["source_view_origin"] == "verified_upstream_restoration"
    assert item["source"]["original_text"] == FULL
    assert item["original_record"] == before[1]
    assert records == before
    assert item["record_blocked"] is False


def test_wrong_icd_uri_or_code_never_restores():
    records, upstream = inputs()
    for key in ("id", "code"):
        wrong = copy.deepcopy(upstream)
        wrong["entities"][0][key] = "different"
        assert not source_restorations(records, wrong, [0,0])


def test_partial_or_nonunique_match_never_restores():
    records, upstream = inputs()
    partial = copy.deepcopy(upstream)
    partial["entities"][0]["diagnosticCriteria"] = FULL.replace("Condition B", "Condition C")
    assert not source_restorations(records, partial, [0,0])
    upstream["entities"][0]["diagnosticCriteria"] = FULL + "\n" + FULL
    assert not source_restorations(records, upstream, [0,0])


def test_no_truncation_flag_does_not_expand_normal_sources():
    records, upstream = inputs()
    records[1]["messages"][1]["content"] = "Medical text:\nFeature A accompanies Condition B."
    assert not source_restorations(records, upstream, [0,0])
