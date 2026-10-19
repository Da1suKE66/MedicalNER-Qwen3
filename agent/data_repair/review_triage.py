"""Build an offline, lossless REVIEW sidecar, never labels or training data.

Proposal-only. Only four release files are read: manifest, records, frozen_split, review_queue.
The train-only surface candidate builder has no file access and rejects holdouts.
All decisions remain PENDING, including mechanically located surface variants.
Offsets are half-open Python Unicode code points in source.original_text.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

from .text_normalization import find_mentions, map_range, normalize_with_offsets


VERSION = "review-triage-20260928-v1"
INPUT_NAMES = ("manifest.json", "records.jsonl", "frozen_split.jsonl", "review_queue.jsonl")
GROUPS = {
    "entity_exact_surface": ("实体：完整词边界精确表面形式", "核对位置；不据此确认医学含义。"),
    "entity_unicode_surface": ("实体：Unicode 等价表面形式", "核对 NFKC 双坐标；不改名称或标签。"),
    "entity_case_whitespace_surface": ("实体：大小写／空白表面形式", "核对完整词边界与原文位置。"),
    "entity_hyphen_boundary_surface": ("实体：连字符／空格分词边界", "词序与字母完全保留；仅审阅表面形式。"),
    "entity_comma_surface": ("实体：仅逗号分隔变体", "词序与所有词保留；仍须审定修饰范围。"),
    "entity_mixed_separator_surface": ("实体：逗号及连字符分隔变体", "核验两类分隔变化；不推断语义等价。"),
    "entity_unsafe_delimiter_review": ("实体：符号／标点范围不确定", "不得删除数字标点、斜杠、括号、否定或句界。"),
    "entity_inside_word_review": ("实体：仅词内命中，不是完整实体定位", "核对词缀、单复数或缺失空格；禁止放宽全局词边界。"),
    "entity_partial_prefix_review": ("实体：仅名称前缀有原文锚点", "补全限定词／分型／原始上下文；前缀不是别名。"),
    "entity_unlocated_review": ("实体：名称／上下文仍未定位", "查原始上下文或人工别名证据；不得判负例。"),
    "relation_evidence_review": ("关系：引用来源／范围／绑定待审", "核验所有证据片段及端点；不猜补引文。"),
    "normality_semantics_review": ("语义：正常性与诊断标准", "人工审定正常人群描述与诊断要求，保留原关系。"),
    "endpoint_type_review": ("语义：端点类型／粒度", "人工审定类型与概念粒度；不修改 ontology。"),
    "normality_hint_review": ("提示：正常性章节", "仅原有 hint，不新增隔离，也不自动移除提示。"),
    "source_json_review": ("源格式：未解析 JSON 块", "完整保留残块；获取同源完整内容，不猜补字段。"),
    "source_boundary_review": ("源格式：句界空格提示", "核查既有证据定位；不重复计算为新修复。"),
    "conflicting_input_review": ("语义：同输入冲突", "补主病种／分型上下文，不合并分型，不解除隔离。"),
    "other_review": ("其他：保留原项待审", "未知规则 fail closed，保持 PENDING。"),
}
MECHANICAL = frozenset(key for key in GROUPS if key.endswith("_surface"))
CODE_GROUPS = {
    "normal_population_as_diagnostic_criterion": "normality_semantics_review",
    "relation_type_reference_requires_review": "endpoint_type_review",
    "normality_section_only_hint": "normality_hint_review",
    "unparsed_json_block": "source_json_review",
    "no_space_sentence_boundary": "source_boundary_review",
    "conflicting_identical_input": "conflicting_input_review",
}
# Deliberately NOT all Unicode punctuation. Minus, slash, apostrophe, decimal,
# sentence boundaries and brackets can change meaning. Never join two words.
HYPHENS = frozenset("-\u2010\u2011")
SEPARATORS = HYPHENS | {","}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode("utf-8")).hexdigest()


def strict_json(value):
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = item
        return result

    def constant(value):
        raise ValueError(f"nonfinite JSON number: {value}")

    return json.loads(value, object_pairs_hook=pairs, parse_constant=constant)


def jsonl(data):
    lines = data.splitlines(keepends=True)
    if any(not line.strip() for line in lines):
        raise ValueError("blank JSONL line would break queue identity")
    rows = [strict_json(line) for line in lines]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("JSONL rows must be objects")
    return rows, lines


def _word(char):
    return char == "_" or char.isalnum() or unicodedata.category(char).startswith("M")


def _tokens(text):
    result, start = [], None
    for i, char in enumerate(text + " "):
        if _word(char):
            if start is None:
                start = i
        elif start is not None:
            result.append((text[start:i], start, i))
            start = None
    return result


def _fold(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _span(raw, start, end):
    a, b = max(0, start - 100), min(len(raw), end + 100)
    return {"start": start, "end": end, "text": raw[start:end],
            "offset_basis": "source.original_text", "offset_unit": "unicode_codepoint",
            "context_start": a, "context_end": b, "context": raw[a:b]}


def _safe_gaps(query, surface):
    """Prove a limited lexical equivalence, NOT a medical alias equivalence."""
    left, right = _tokens(query), _tokens(surface)
    if [t[0].casefold() for t in left] != [t[0].casefold() for t in right] or not left:
        return None
    # All query punctuation must be between words, never discarded at the ends.
    if query[:left[0][1]].strip() or query[left[-1][2]:].strip():
        return None
    changes, kinds = [], set()
    for i in range(len(left) - 1):
        before = query[left[i][2]:left[i + 1][1]]
        after = surface[right[i][2]:right[i + 1][1]]
        if _fold(before) == _fold(after):
            continue
        # Even a comma between 1 and 000 or a hyphen around an alphanumeric code
        # must not be treated as typographic evidence. No newline/paragraph join.
        adjacent = (left[i][0], left[i + 1][0], right[i][0], right[i + 1][0])
        if (not before or not after or any(not word.isalpha() for word in adjacent)
                or any(c in "\r\n" for c in before + after)
                or any(not c.isspace() and c not in SEPARATORS for c in before + after)):
            return None
        punctuation = {c for c in before + after if not c.isspace()}
        if not punctuation:
            continue
        # Repeated punctuation, or a comma-to-hyphen substitution, is ambiguous.
        if any(sum(not c.isspace() for c in gap) > 1 for gap in (before, after)):
            return None
        if "," in punctuation and punctuation & HYPHENS:
            return None
        kinds.add("comma" if "," in punctuation else "hyphen")
        changes.append({"after_token": i + 1, "before": before, "after": after})
    if not changes:
        return None
    return kinds, changes


def classify_entity(name, raw):
    """Return reproducible local string observations; do not use another record."""
    if not isinstance(name, str) or not name.strip() or not isinstance(raw, str):
        raise ValueError("invalid entity name or source")
    strict = find_mentions(name, raw)
    if strict:
        if any(raw[a:b] == name for a, b in strict):
            group = "entity_exact_surface"
        elif unicodedata.normalize("NFKC", name) != name or any(
            unicodedata.normalize("NFKC", raw[a:b]) != raw[a:b] for a, b in strict
        ):
            group = "entity_unicode_surface"
        else:
            group = "entity_case_whitespace_surface"
        return group, {"matches": [_span(raw, a, b) for a, b in strict],
                       "rule": "symmetric_nfkc_case_whitespace_with_complete_word_boundaries"}

    # repair=False: no dictionary, medical spelling fix, synonym or stemming.
    mapping = normalize_with_offsets(raw, repair=False)
    text = mapping["text"]
    query = unicodedata.normalize("NFKC", name).strip()
    wanted, tokens = _tokens(query), _tokens(text)
    words = [t[0].casefold() for t in wanted]
    observed = [t[0].casefold() for t in tokens]
    safe, unsafe, kinds = [], [], set()
    for i in range(len(tokens) - len(words) + 1) if words else ():
        if observed[i:i + len(words)] != words:
            continue
        start, end = tokens[i][1], tokens[i + len(words) - 1][2]
        a, b = map_range(mapping, start, end)
        hit = _span(raw, a, b)
        proof = _safe_gaps(query, text[start:end])
        if proof:
            kinds.update(proof[0])
            hit["separator_changes"] = proof[1]
            safe.append(hit)
        else:
            unsafe.append(hit)
    if safe:
        group = {frozenset({"comma"}): "entity_comma_surface",
                 frozenset({"hyphen"}): "entity_hyphen_boundary_surface"}.get(
                     frozenset(kinds), "entity_mixed_separator_surface")
        return group, {"matches": safe, "rule": "same_complete_words_same_order_allowlisted_separators",
                       "unsafe_lexical_matches": unsafe}
    if unsafe:
        return "entity_unsafe_delimiter_review", {"matches": [], "lexical_hints": unsafe,
                                                  "rule": "unsafe_delimiters_not_equivalence"}

    # A diagnostic only. In particular Manic in Hypomanic is NOT a mention.
    inside = []
    for match in re.finditer(re.escape(query), text, flags=re.IGNORECASE):
        a, b = map_range(mapping, *match.span())
        hit = _span(raw, a, b)
        hit["left_word_continuation"] = bool(match.start() and _word(text[match.start() - 1]))
        hit["right_word_continuation"] = bool(match.end() < len(text) and _word(text[match.end()]))
        inside.append(hit)
    if inside:
        return "entity_inside_word_review", {"matches": [], "lexical_hints": inside,
                                             "rule": "substring_only_never_relax_boundaries"}
    # Longest literal prefix only; no word dropping, reordering or semantic lookup.
    for size in range(len(wanted) - 1, 0, -1):
        prefix = query[:wanted[size - 1][2]]
        hits = find_mentions(prefix, raw)
        if hits:
            return "entity_partial_prefix_review", {
                "matches": [], "prefix": prefix, "unmatched_suffix": query[wanted[size - 1][2]:],
                "lexical_hints": [_span(raw, a, b) for a, b in hits],
                "rule": "literal_prefix_is_not_full_entity_or_alias",
            }
    return "entity_unlocated_review", {"matches": [], "lexical_hints": [],
                                        "rule": "no_accepted_local_surface_evidence"}


def _focus(issue, record):
    graph = record["repaired_graph"]
    nodes = graph["entities"]
    details = issue.get("details", {})
    entity_id = issue.get("entity_id")
    relation_index = issue.get("relation_index", details.get("relation_index"))
    selected = set()
    whole_record = issue["kind"] not in {"entity", "relation"} and relation_index is None
    if issue["kind"] == "relation" and relation_index is None:
        raise ValueError("missing relation reference")
    if issue["kind"] == "entity":
        selected = {i for i, edge in enumerate(graph["relations"])
                    if str(entity_id) in (str(edge["source"]), str(edge["target"]))}
    elif relation_index is not None:
        if type(relation_index) is not int or not 0 <= relation_index < len(graph["relations"]):
            raise ValueError("missing relation reference")
        selected = {relation_index}
    elif whole_record:
        selected = set(range(len(graph["relations"])))
    relations = [{"index": i, "annotation": edge} for i, edge in enumerate(graph["relations"])
                 if i in selected]
    endpoints = {str(edge[k]) for i, edge in enumerate(graph["relations"]) if i in selected
                 for k in ("source", "target")}
    if entity_id is not None:
        endpoints.add(str(entity_id))
    focus = {"entities": [node for node in nodes if whole_record or str(node["id"]) in endpoints],
             "relations": relations,
             "existing_evidence_audit": [a for a in record["annotations"] if a["index"] in selected],
             "existing_record_issues": record.get("issues", []),
             "record_blocked": record["record_blocked"], "blocked_pairs": record["blocked_pairs"]}
    if issue["kind"] == "entity":
        targets = [node for node in nodes if str(node["id"]) == str(entity_id)]
        if len(targets) != 1:
            raise ValueError("missing or duplicate entity reference")
        focus["entity"] = targets[0]
        focus["existing_entity_audit"] = [n for n in record["entities"] if str(n["id"]) == str(entity_id)]
    return deepcopy(focus)


def triage_issue(issue, record, *, queue_sha256, queue_line, snapshot_line):
    if issue["split"] != record["split"] or issue["source_index"] != record["source_index"]:
        raise ValueError("issue/record split or identity mismatch")
    focus = _focus(issue, record)
    if issue["kind"] == "entity":
        node = focus["entity"]
        name = node.get("name") or node.get("properties", {}).get("Name")
        category, observations = classify_entity(name, record["source"]["original_text"])
    elif issue["kind"] == "relation":
        category, observations = "relation_evidence_review", {}
    else:
        category = CODE_GROUPS.get(issue.get("details", {}).get("code"), "other_review")
        observations = {}
    identifier = f"rq-{queue_sha256}-L{queue_line:06d}"
    return {
        "triage_id": identifier,
        "issue_id": deepcopy(issue.get("issue_id", issue.get("id", identifier))),
        "original_issue": deepcopy(issue), "original_issue_sha256": sha(canonical(issue)),
        "queue_line": queue_line, "queue_sha256": queue_sha256,
        "source_index": record["source_index"], "split": record["split"], "group_id": record["group_id"],
        "record_ref": {"file": "source_records.jsonl", "line": snapshot_line,
                       "prepared_record_sha256": sha(canonical(record)),
                       "original_record_sha256": record["original_record_sha256"],
                       "source_original_sha256": record["source"]["original_sha256"]},
        "category": category, "review_guidance": GROUPS[category][1],
        "proposal_only": True,
        "mechanical_surface_located": category in MECHANICAL,
        "separator_scope_requires_human_review": category in {
            "entity_comma_surface", "entity_hyphen_boundary_surface", "entity_mixed_separator_surface",
            "entity_unsafe_delimiter_review"},
        "observations": observations, "focus": focus,
        "decision": "PENDING", "human_review_required": True,
        "not_a_negative": True, "gold": False, "label_change": None,
        "semantic_equivalence_confirmed": False, "automatic_training_use_allowed": False,
    }


def build_train_surface_candidates(train_items):
    """No I/O; train-only, unapproved spelling observations, not medical aliases.

    Reject the whole input on any holdout *before* examining semantic fields.
    No label is transferred across records or across splits.
    """
    items = list(train_items)
    if any(item["split"] != "train" for item in items):
        raise ValueError("train-only surface builder rejects tune/test/other splits")
    grouped = defaultdict(list)
    for item in items:
        if not item["mechanical_surface_located"]:
            continue
        if item["category"] not in MECHANICAL or not item["observations"].get("matches"):
            raise ValueError("invalid mechanical surface observation")
        node = item["focus"]["entity"]
        name = node.get("name") or node.get("properties", {}).get("Name")
        for hit in item["observations"]["matches"]:
            grouped[(name, node["label"], hit["text"])].append({
                "triage_id": item["triage_id"], "source_index": item["source_index"],
                "entity_id": node["id"], "split": "train", "start": hit["start"], "end": hit["end"],
                "source_original_sha256": item["record_ref"]["source_original_sha256"],
            })
    return [{"canonical_name": key[0], "unchanged_entity_label": key[1], "observed_surface": key[2],
             "provenance": provenance, "split_scope": "train_only", "status": "PENDING_REVIEW",
             "proposal_only": True, "is_medical_alias": False, "automatic_training_use_allowed": False}
            for key, provenance in sorted(grouped.items())]


def _validate_inputs(manifest, records, frozen, queue, expected_items):
    if len(queue) != expected_items or manifest.get("review_items") != len(queue):
        raise ValueError("review item count mismatch")
    if manifest.get("summary", {}).get("records") != len(records):
        raise ValueError("record count mismatch")
    indexed, groups = {}, {}
    for record in records:
        index, split = record["source_index"], record["split"]
        if type(index) is not int or index < 0 or index in indexed:
            raise ValueError("duplicate or invalid source_index")
        if split not in {"train", "tune", "test"}:
            raise ValueError("unrecognized split")
        group = canonical(record["group_id"])
        if group in groups and groups[group] != split:
            raise ValueError("cross-split source group")
        groups[group], indexed[index] = split, record
        source = record["source"]
        checks = ((record["original_record_sha256"], sha(canonical(record["original_record"]))),
                  (source["original_sha256"], sha(source["original_text"])),
                  (source["sha256"], sha(source["text"])))
        if any(a != b for a, b in checks):
            raise ValueError("record/source hash mismatch")
        ids = [str(node["id"]) for node in record["repaired_graph"]["entities"]]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate entity ID")
        if any(str(edge[k]) not in ids for edge in record["repaired_graph"]["relations"]
               for k in ("source", "target")):
            raise ValueError("dangling relation endpoint")
    splits = {}
    for entry in frozen:
        index = entry["source_index"]
        if index in splits:
            raise ValueError("duplicate frozen source_index")
        splits[index] = entry
    if set(splits) != set(indexed):
        raise ValueError("frozen split coverage mismatch")
    for index, record in indexed.items():
        if any(record[k] != splits[index][k] for k in ("group_id", "split")):
            raise ValueError("frozen split mismatch")
    for issue in queue:
        index = issue["source_index"]
        if type(index) is not int or index not in indexed:
            raise ValueError("missing issue source reference")
        if issue["split"] != indexed[index]["split"]:
            raise ValueError("issue split mismatch")
        if issue.get("decision") != "PENDING" or issue.get("not_a_negative") is not True:
            raise ValueError("expected a pending non-negative review item")
    return indexed


def _fence(value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    longest = max((len(m.group()) for m in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}json\n{text}\n{fence}\n"


def render_group(category, items):
    lines = [f"# {GROUPS[category][0]}\n", GROUPS[category][1] + "\n",
             f"{len(items)} 项；全部 proposal-only / PENDING / human-review。不是金标，也不是训练输入。"
             "纯标点变化也可能改变语义／修饰范围；没有任何自动语义等价裁定。\n",
             "原始记录完整保存在 [source_records.jsonl](../source_records.jsonl)，逐项引用行号及哈希。\n"]
    for item in items:
        lines.extend([f"## {item['triage_id']}\n",
                      f"source_index={item['source_index']}；split={item['split']}；"
                      f"原记录快照行={item['record_ref']['line']}；旧队列行={item['queue_line']}。\n",
                      "### 原始问题与理由（未改写）\n", _fence(item["original_issue"]),
                      "### 字符定位／词法提示（不是语义裁定）\n", _fence(item["observations"]),
                      "### 原实体、关联关系与原证据审核状态（全部保留）\n", _fence(item["focus"]),
                      "人工记录：审阅人／日期／证据坐标／范围／结论／版本：**待填写**。"
                      "本包不导入人工结论，不改变原监督。\n"])
    return "\n".join(lines)


def run(candidate_dir, output_dir, *, expected_items=264, expected_manifest_sha256=None):
    candidate = Path(candidate_dir).resolve(strict=True)
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise ValueError("refusing to overwrite an existing review directory")
    output = output.resolve()
    if candidate == output or candidate in output.parents or output in candidate.parents:
        raise ValueError("output must be separate from the candidate")
    inputs = {}
    for name in INPUT_NAMES:
        path = candidate / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"expected regular input file: {name}")
        inputs[name] = path.read_bytes()
    hashes = {name: sha(data) for name, data in inputs.items()}
    if expected_manifest_sha256 is not None and hashes["manifest.json"] != expected_manifest_sha256:
        raise ValueError("unexpected manifest SHA256")
    manifest = strict_json(inputs["manifest.json"])
    for name in INPUT_NAMES[1:]:
        if manifest.get("output_sha256", {}).get(name) != hashes[name]:
            raise ValueError(f"candidate file hash mismatch: {name}")
    records, record_lines = jsonl(inputs["records.jsonl"])
    frozen, _ = jsonl(inputs["frozen_split.jsonl"])
    queue, _ = jsonl(inputs["review_queue.jsonl"])
    indexed = _validate_inputs(manifest, records, frozen, queue, expected_items)
    needed = {item["source_index"] for item in queue}
    snapshots = [(r, line) for r, line in zip(records, record_lines) if r["source_index"] in needed]
    snapshot_lines = {r["source_index"]: i for i, (r, _) in enumerate(snapshots, 1)}
    items = [None] * len(queue)

    def classify(i):
        issue = queue[i]
        return triage_issue(issue, indexed[issue["source_index"]], queue_sha256=hashes["review_queue.jsonl"],
                            queue_line=i + 1, snapshot_line=snapshot_lines[issue["source_index"]])

    # Complete the train-only table BEFORE classifying holdouts. The alias/surface
    # builder never receives or reads tune/test records, targets, names or evidence.
    for i, issue in enumerate(queue):
        if issue["split"] == "train":
            items[i] = classify(i)
    train_candidates = build_train_surface_candidates(item for item in items if item is not None)
    for i, item in enumerate(items):
        if item is None:
            items[i] = classify(i)
    categories = Counter(item["category"] for item in items)
    mechanical_count = sum(item["mechanical_surface_located"] for item in items)
    counts = {
        "input_records": len(records), "preserved_review_records": len(snapshots),
        "input_issues": len(queue), "output_issues": len(items),
        "categories": {k: categories[k] for k in GROUPS},
        "by_kind": dict(sorted(Counter(q["kind"] for q in queue).items())),
        "by_split": dict(sorted(Counter(q["split"] for q in queue).items())),
        "by_category_and_split": {k: dict(sorted(Counter(i["split"] for i in items if i["category"] == k).items()))
                                  for k in GROUPS},
        "mechanically_located_entity_issues": mechanical_count,
        "still_unlocated_entity_issues": sum(q["kind"] == "entity" for q in queue) - mechanical_count,
        "nonmechanical_issues": len(items) - mechanical_count,
        "human_review_required": len(items), "automatically_closed": 0, "label_changes": 0,
        "train_surface_candidate_rows": len(train_candidates),
    }
    if (sum(categories.values()) != len(queue) or len({i["triage_id"] for i in items}) != len(queue)
            or [i["original_issue"] for i in items] != queue):
        raise RuntimeError("review conservation failed")
    # Fail before creating output if an input changed during classification.
    if any(sha((candidate / name).read_bytes()) != digest for name, digest in hashes.items()):
        raise ValueError("source changed during triage")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    output_hashes = {}

    def write(name, value):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        data = value if isinstance(value, bytes) else value.encode("utf-8")
        with path.open("xb") as handle:
            path.chmod(0o600)
            handle.write(data)
        output_hashes[name] = sha(data)

    write("original_review_queue.jsonl", inputs["review_queue.jsonl"])
    write("source_records.jsonl", b"".join(line if line.endswith(b"\n") else line + b"\n"
                                           for _, line in snapshots))
    write("triage.jsonl", "".join(canonical(item) + "\n" for item in items))
    write("train_surface_candidates.jsonl", "".join(canonical(item) + "\n" for item in train_candidates))
    readme = ["# 264 项复核清单分流／审阅包\n" if len(items) == 264 else "# 复核清单分流／审阅包\n",
              f"输入 {len(queue)} 项 → 输出 {len(items)} 项。"
              f"human-review 仍需 **{len(items)} 项**，自动关闭 **0**，标签变更 **0**。\n",
              f"其中 {mechanical_count} 项只确认了机械表面形式／字符位置；"
              f"其余 {len(items) - mechanical_count} 项仍有非机械疑问。"
              "两类都保持 proposal-only / PENDING。纯标点也可能改变语义／修饰范围，必须留审。"
              "这不是金标、语义别名字典、负例池或训练发布。\n",
              "源记录／隔离／UNKNOWN 状态未改变；未定位不等于假实体。"
              "ICD 未命名列表等不在旧 264 项内的问题不计入本包完成度，仍保留在原候选 records 中；"
              "本包只快照与旧复核项关联的记录。\n",
              "## 文件与复现\n",
              "[triage.jsonl](triage.jsonl) 按旧队列原顺序逐项保留 ID、理由、关联证据和快照引用；"
              "[original_review_queue.jsonl](original_review_queue.jsonl) 是旧队列逐字节副本。\n",
              "[source_records.jsonl](source_records.jsonl) 保存所涉及记录的完整原记录、修复视图、"
              "原文、来源列表及历史证据；record_ref 的行号从 1 开始，source_index 保持原值。\n",
              "[train_surface_candidates.jsonl](train_surface_candidates.jsonl) 仅含训练项的机械表面形式候选。"
              "只在 train 内构建，不读取 tune/test 训练文件、不跨 split 套用、不声称医学别名。\n",
              "[manifest.json](manifest.json) 记录输入／输出／代码哈希、计数和安全门禁。"
              "同输入与代码产生相同字节；输出目录存在即拒绝。\n",
              "## 分组\n", "| 分组 | 项数 |\n|---|---:|"]
    for category, (title, _) in GROUPS.items():
        group_items = [item for item in items if item["category"] == category]
        if group_items:
            write(f"groups/{category}.md", render_group(category, group_items))
            readme.append(f"| [{title}](groups/{category}.md) | {len(group_items)} |")
        else:
            readme.append(f"| {title} | 0 |")
    readme.append("\n审阅结果请另存带审阅人、日期、证据坐标与来源哈希的新版本；勿覆盖本包或旧候选。\n")
    write("README.md", "\n".join(readme))
    module = Path(__file__).resolve()
    result = {
        "version": VERSION, "status": "PROPOSAL_ONLY_PENDING_HUMAN_REVIEW_NOT_GOLD", "counts": counts,
        "proposal_only": True,
        "input_sha256": hashes, "upstream_teacher_sha256": manifest.get("source_sha256"),
        "code_sha256": {p.name: sha(p.read_bytes()) for p in (module, module.with_name("text_normalization.py"))},
        "runtime": {"python": sys.version.split()[0], "unicode": unicodedata.unidata_version},
        "output_sha256": dict(output_hashes),
        "invariants": {"queue_conserved": True, "original_queue_byte_identical": True,
                       "records_preserved_by_reference_and_full_snapshot": True, "input_hashes_rechecked": True,
                       "frozen_split_validated": True, "all_pending": True, "all_not_a_negative": True},
        "gates": {"gold": False, "human_semantic_adjudication": False,
                  "medical_alias_confirmed": False, "automatic_training_use_allowed": False,
                  "labels_modified": False, "negative_examples_created": False},
        "train_surface_policy": {"allowed_split": "train", "holdouts_rejected_by_builder": True,
                                 "external_alias_sources": [], "cross_split_application": False,
                                 "training_or_evaluation_files_read": []},
        "scope": "Only the existing queue; not a complete medical/source audit or a label repair.",
    }
    write("manifest.json", json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-items", type=int, default=264)
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    result = run(args.candidate_dir, args.output_dir, expected_items=args.expected_items,
                 expected_manifest_sha256=args.expected_manifest_sha256)
    print(canonical({"version": result["version"], "counts": result["counts"], "gates": result["gates"]}))


if __name__ == "__main__":
    main()
