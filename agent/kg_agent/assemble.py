"""Deterministic final JSON assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import Graph, natural_id_key


@dataclass(frozen=True)
class AssemblerConfig:
    evidence_key: str = "evidence"
    evidence_as_list: bool = False
    include_provenance: bool = False
    include_empty_properties: bool = True


def _evidence_value(values: list[str], as_list: bool) -> Any:
    if as_list:
        return list(values)
    if not values:
        return ""
    return values[0] if len(values) == 1 else list(values)


def assemble_graph(graph: Graph, config: AssemblerConfig | None = None) -> dict[str, list[dict[str, Any]]]:
    config = config or AssemblerConfig()
    entities: list[dict[str, Any]] = []
    for entity in sorted(graph.entities, key=lambda item: natural_id_key(item.id)):
        properties = {"Name": entity.name, **entity.properties}
        payload: dict[str, Any] = {
            "id": entity.id,
            "label": entity.label,
            "name": entity.name,
            "properties": properties if (properties or config.include_empty_properties) else {},
        }
        if config.include_provenance:
            payload["provenance"] = {
                "span_ids": list(entity.span_ids),
                "importance": entity.importance,
            }
        entities.append(payload)

    relations: list[dict[str, Any]] = []
    for relation in sorted(
        graph.relations,
        key=lambda item: (natural_id_key(item.source), item.relation, natural_id_key(item.target)),
    ):
        payload = {
            "source": relation.source,
            "target": relation.target,
            "relation": relation.relation,
            config.evidence_key: _evidence_value(relation.evidence_text, config.evidence_as_list),
        }
        if config.include_provenance:
            payload["evidence_span_ids"] = list(relation.evidence_ids)
        payload.update(relation.properties)
        relations.append(payload)
    return {"entities": entities, "relations": relations}
