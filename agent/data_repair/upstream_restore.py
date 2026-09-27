"""Recover a cut sentence only from a uniquely matched, identity-bound source.

Never infer a missing word. Both the COMPLETE surviving chunk and the exact ICD
release URI must match an original field. Failure yields no automatic recovery.
"""

from collections import defaultdict
import hashlib

from build_stage_data import _message
from kg_agent.source import SentenceSpanSplitter
from .source_provenance import source_view
from .text_normalization import map_range, normalize_with_offsets


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def source_restorations(records, upstream, groups):
    nodes = upstream.get("entities", []) if isinstance(upstream, dict) else []
    by_uri = defaultdict(list)
    for index, node in enumerate(nodes):
        if isinstance(node, dict) and isinstance(node.get("id"), str):
            by_uri[node["id"]].append((index, node))
    anchors, results = {}, {}
    for index, (record, group) in enumerate(zip(records, groups)):
        user = _message(record, "user")
        view = source_view(user)
        if view["source_type"] == "structured_icd":
            metadata = view["routed"]["metadata"]
            anchors[group] = (index, metadata.get("uri"), metadata.get("code"))
            continue
        if not any(issue["code"] == "suspected_start_truncation" for issue in view["issues"]):
            continue
        anchor = anchors.get(group)
        if not anchor or len(by_uri.get(anchor[1], [])) != 1:
            continue
        upstream_index, node = by_uri[anchor[1]][0]
        if node.get("code") != anchor[2]:
            continue
        query = normalize_with_offsets(view["raw_payload"])["text"]
        matches = []
        for field in ("diagnosticCriteria", "definition", "longDefinition"):
            original = node.get(field)
            if not isinstance(original, str) or not original:
                continue
            mapping = normalize_with_offsets(original)
            document = mapping["text"]
            position = document.find(query)
            if position < 0 or document.find(query, position + 1) >= 0:
                continue
            # Query must start strictly inside an existing original sentence.
            ranges = [(a,b) for a,b in SentenceSpanSplitter.sentence_ranges(document)
                      if a < position < b]
            if len(ranges) != 1:
                continue
            start = ranges[0][0]
            a,b = map_range(mapping, start, position)
            prefix = original[a:b]
            # A cut inside a normalization expansion has no exact reversible
            # boundary. This equality also rejects guessed/overlapping prefixes.
            if normalize_with_offsets(prefix + view["raw_payload"])["text"] != document[start:position] + query:
                continue
            matches.append({"upstream_record_index": upstream_index, "field": field,
                            "field_sha256": sha(original), "upstream_prefix_start": a,
                            "upstream_prefix_end": b, "prefix": prefix,
                            "full_chunk_normalized_start": position,
                            "full_chunk_normalized_end": position + len(query),
                            "full_chunk_unique_match": True})
        if len(matches) != 1:
            continue
        evidence = matches[0]
        replacement = user[:view["payload_start"]] + evidence["prefix"] + view["raw_payload"]
        results[index] = {"source_index": index, "group_id": group,
                          "method": "exact_full_chunk_and_same_icd_release_uri_prepend_original_sentence",
                          "anchor_source_index": anchor[0], "anchor_uri": anchor[1], "anchor_code": anchor[2],
                          "original_user_sha256": sha(user), "effective_user_sha256": sha(replacement),
                          "effective_user": replacement, **evidence}
    return results
