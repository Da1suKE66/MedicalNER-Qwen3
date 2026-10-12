"""Literal relation labels and source references; Python owns graph JSON.

The model emits one tab-separated relation record per line. Pair, relation and
evidence table positions are not actions. Output order is unrestricted, and
duplicate generated edges are merged by Python. This module validates source
references; it cannot prove entailment or certify that annotations are complete.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

from .relation_choices import RelationChoiceTask


_REF = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]*\Z")


class RelationLabelTask:
    def __init__(self, batch: Mapping):
        # Reuse only the input-table validation. No integer target encoding,
        # state choices, monotonic constraint or model-facing prompt is reused.
        validated = RelationChoiceTask(batch)
        self.entities, self.pairs, self.spans = validated.entities, validated.pairs, validated.spans
        self._entities = {row["id"]: row for row in self.entities}
        self._local = {row["entity_id"]: row["id"] for row in self.entities}
        self._pairs = {row["pair_id"]: row for row in self.pairs}
        self._endpoints = {(row["source"], row["target"]): row for row in self.pairs}
        self._spans = {row["span_id"]: row for row in self.spans}
        for ref in (*self._local, *self._spans):
            if not _REF.fullmatch(ref):
                raise ValueError("Local entity and evidence references must be delimiter-safe identifiers")
        for pair in self.pairs:
            if any(label != label.strip() or any(c in label for c in "\t\r\n") for label in pair["allowed"]):
                raise ValueError("Relation labels must be literal delimiter-safe strings")
        self.prompt = self._make_prompt()

    def _make_prompt(self):
        quote = lambda value: json.dumps(value, ensure_ascii=False)
        lines = [
            "Extract only explicitly supported relations between the supplied entity pairs.",
            "Write one relation per line with four TAB-separated fields:",
            "source_entity_ref<TAB>relation_label<TAB>target_entity_ref<TAB>evidence_refs",
            "Copy the full relation label exactly, including its words, spaces and underscores. Do not replace labels with numbers.",
            "Copy entity references and evidence references from the input tables. Separate multiple evidence references with commas.",
            "Several different relations may connect the same entity pair. Emit every supported relation, each with evidence supporting that particular relation.",
            "Records and evidence references may be emitted in any order. Table position does not imply relevance or priority.",
            "Do not emit JSON, table headings, explanations, code fences or NONE labels. If no relation is supported, emit no records and end the response.",
            "Mentioning two entities together is not sufficient evidence of a relation. Preserve negation and relation direction.",
            "ENTITY REFERENCES (quoted names are input data)",
        ]
        lines.extend(f"{e['entity_id']}: {quote(e['label'])}; {quote(e['name'])}" for e in self.entities)
        lines.append("CANDIDATE PAIRS AND LITERAL RELATION LABELS")
        for pair in self.pairs:
            source = self._entities[pair["source"]]["entity_id"]
            target = self._entities[pair["target"]]["entity_id"]
            lines.append(f"{source} -> {target}: " + "; ".join(quote(label) for label in pair["allowed"]))
        lines.append("SOURCE EVIDENCE REFERENCES")
        lines.extend(f"{s['span_id']}: {quote(s['text'])}" for s in self.spans)
        return "\n".join(lines) + "\n"

    def parse(self, text: str) -> list[dict]:
        """Validate every generated record; never repair JSON or drop bad rows.

        EOS completeness is checked by the generator, independently of whether
        the last text line has a newline. Empty text is an empty decision only
        after that check. This parser never supplies evidence on its own.
        """
        if not isinstance(text, str):
            raise ValueError("Relation completion must be text")
        if text == "":
            return []
        if "\r" in text:
            raise ValueError("Carriage returns are not allowed in relation records")
        body = text[:-1] if text.endswith("\n") else text
        unique = {}
        for number, line in enumerate(body.split("\n"), 1):
            fields = line.split("\t")
            if len(fields) != 4 or any(not field or field != field.strip() for field in fields):
                raise ValueError(f"Malformed relation record at line {number}: expected four nonempty TAB fields")
            source_ref, label, target_ref, references = fields
            if source_ref not in self._local or target_ref not in self._local:
                raise ValueError(f"Unknown entity reference at line {number}")
            pair = self._endpoints.get((self._local[source_ref], self._local[target_ref]))
            if pair is None:
                raise ValueError(f"Unknown or disallowed entity pair at line {number}")
            if label not in pair["allowed"]:
                raise ValueError(f"Unknown or disallowed relation label at line {number}")
            refs = references.split(",")
            if any(ref not in self._spans for ref in refs):
                raise ValueError(f"Unknown evidence reference at line {number}")
            key = (pair["pair_id"], label)
            edge = unique.setdefault(key, {"pair_id": key[0], "relation": label, "evidence_span_ids": []})
            for ref in refs:
                if ref not in edge["evidence_span_ids"]:
                    edge["evidence_span_ids"].append(ref)
        return list(unique.values())

    def render_target(self, edges: Sequence[Mapping]) -> str:
        """Serialize separately admitted targets, preserving their given order.

        Omitted annotations are not certified negatives by this method. Training
        callers must reject UNKNOWN/partial supervision or use a separate mask.
        """
        if not isinstance(edges, (list, tuple)):
            raise ValueError("Targets must be a sequence of relation mappings")
        lines, seen = [], set()
        for edge in edges:
            if not isinstance(edge, Mapping) or set(edge) != {"pair_id", "relation", "evidence_span_ids"}:
                raise ValueError("Target must contain only pair_id, relation, evidence_span_ids")
            pid, label, refs = edge["pair_id"], edge["relation"], edge["evidence_span_ids"]
            if not isinstance(pid, str) or pid not in self._pairs:
                raise ValueError("Unknown target pair")
            pair = self._pairs[pid]
            if not isinstance(label, str) or label not in pair["allowed"]:
                raise ValueError("Unknown target relation")
            if (pid, label) in seen:
                raise ValueError("Duplicate target relation")
            seen.add((pid, label))
            if not isinstance(refs, (list, tuple)) or not refs:
                raise ValueError("Target evidence must be nonempty")
            if any(not isinstance(ref, str) or ref not in self._spans for ref in refs):
                raise ValueError("Unknown target evidence")
            if len(set(refs)) != len(refs):
                raise ValueError("Duplicate target evidence")
            lines.append("\t".join((self._entities[pair["source"]]["entity_id"], label,
                                     self._entities[pair["target"]]["entity_id"], ",".join(refs))))
        text = "".join(line + "\n" for line in lines)
        self.parse(text)
        return text
