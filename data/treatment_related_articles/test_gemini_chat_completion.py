#!/usr/bin/env python3
"""Generate all 22 treatment-related KG samples through a compatible API.

The script makes at most one API request per selected sample. It never retries a
failed request automatically. Successful samples are checkpointed immediately,
so rerunning the script skips completed indices unless --overwrite is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any


# Fill the key here, or set the XI_AI_API_KEY environment variable.
api_key = "sk-SCP0FiLRcsRfCwUX0b4aB1B512E94075913c8f0d2d273b38"
api_base = "https://api-2.xi-ai.cn/v1"
model_name = "gemini-3.1-pro-preview"

FORMAT_ENFORCEMENT = """

Mandatory response contract for this request:
- Return exactly two top-level blocks in this order: <think>...</think> and <output>...</output>.
- Never omit the <think> block, even when the extraction is simple.
- Put one strict JSON object inside <output>; do not use Markdown code fences.
- The JSON object must contain both an entities list and a relations list. Use an empty list when no relation is supported.
""".rstrip()

script_dir = Path(__file__).resolve().parent
dataset_path = script_dir / "treatment_related_articles_predict_22_llamafactory.json"
manifest_path = script_dir / "treatment_related_articles_predict_22_manifest.json"
generated_output_path = (
    script_dir / "treatment_related_articles_gemini_generated_22.json"
)
completed_output_path = (
    script_dir
    / "treatment_related_articles_predict_22_gemini_completed_llamafactory.json"
)
metadata_output_path = (
    script_dir / "treatment_related_articles_gemini_generated_22.metadata.json"
)


ALLOWED_LABELS = {
    "Disease",
    "Symptom",
    "Diagnostic Criteria",
    "Interview Tool",
    "Patient Information",
    "Medication",
    "Communication Method",
    "Risk Information",
}

LABEL_NORMALIZATION = {
    "Core Symptom": "Symptom",
    "Associated Symptom": "Symptom",
    "Differential Diagnosis": "Disease",
    "DSM-5 Diagnostic Criteria": "Diagnostic Criteria",
    "Key Interview Points": "Interview Tool",
    "Patient Features": "Patient Information",
    "Drug": "Medication",
    "Dialogue Strategy": "Communication Method",
    "Risk Factors": "Risk Information",
}

RELATION_CONTRACT = {
    "subsumes": ("Disease Hierarchy", "Subsumes"),
    "differentiates_from": ("Disease Hierarchy", "Differentiates From"),
    "co_occurs_with_frequency": (
        "Disease Hierarchy",
        "Co-occurrence Frequency",
    ),
    "associated_with_poor_prognosis_in": (
        "Disease Hierarchy",
        "Associated with Poor Prognosis",
    ),
    "is_core_symptom_of": ("Symptom-Disease", "Core Symptom Of"),
    "is_associated_symptom_of": (
        "Symptom-Disease",
        "Associated Symptom Of",
    ),
    "precedes": ("Symptom-Disease", "Precedes"),
    "follows": ("Symptom-Disease", "Follows"),
    "modulated_by": ("Symptom-Disease", "Modulated By"),
    "required_for_diagnosis_of": (
        "Diagnostic Criteria-Disease-Symptom Triangle",
        "Required For Diagnosis Of",
    ),
    "excludes_if_present": (
        "Diagnostic Criteria-Disease-Symptom Triangle",
        "Excludes If Present",
    ),
    "supports_subtyping_of": (
        "Diagnostic Criteria-Disease-Symptom Triangle",
        "Supports Subtyping Of",
    ),
    "assesses_for": ("Interview Tool-Symptom-Patient Info", "Assesses For"),
    "triggers_follow_up_question_on": (
        "Interview Tool-Symptom-Patient Info",
        "Triggers Follow-up Question On",
    ),
    "informed_by_patient_demographics": (
        "Interview Tool-Symptom-Patient Info",
        "Informed By Patient Demographics",
    ),
    "first_line_for": ("Medication-Disease-Patient Info", "First Line For"),
    "contraindicated_in": (
        "Medication-Disease-Patient Info",
        "Contraindicated In",
    ),
    "dose_adjusted_for": (
        "Medication-Disease-Patient Info",
        "Dose Adjusted For",
    ),
    "interacts_with": ("Medication-Disease-Patient Info", "Interacts With"),
    "recommended_for": ("Communication-Patient-Risk", "Recommended For"),
    "avoid_with": ("Communication-Patient-Risk", "Avoid With"),
    "escalates_to": ("Communication-Patient-Risk", "Escalates To"),
    "triggers_alert_when": ("Risk-Disease-Symptom", "Triggers Alert When"),
    "mediated_by": ("Risk-Disease-Symptom", "Mediated By"),
}

LABEL_ID_PREFIX = {
    "Disease": "D",
    "Symptom": "S",
    "Diagnostic Criteria": "DC",
    "Interview Tool": "IT",
    "Patient Information": "PI",
    "Medication": "M",
    "Communication Method": "CM",
    "Risk Information": "R",
}

GENERIC_NODE_NAMES = {
    "Disease (DSM-5)",
    "Core Symptom",
    "Associated Symptom",
    "Differential Diagnosis",
    "DSM-5 Diagnostic Criteria",
    "Key Interview Points",
    "Patient Features",
    "Drug",
    "Dialogue Strategy",
    "Risk Factors",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    temporary_path.replace(path)


def load_source_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = load_json(dataset_path)
    manifest = load_json(manifest_path)

    if not isinstance(dataset, list):
        raise ValueError("The source dataset must be a top-level JSON list.")
    manifest_records = manifest.get("records") if isinstance(manifest, dict) else None
    if not isinstance(manifest_records, list):
        raise ValueError("The manifest must contain a records list.")
    if len(dataset) != len(manifest_records):
        raise ValueError(
            f"Dataset/manifest size mismatch: {len(dataset)} != "
            f"{len(manifest_records)}."
        )
    if len(dataset) != 22:
        raise ValueError(f"Expected 22 source samples, found {len(dataset)}.")

    return dataset, manifest_records


def request_messages(sample: dict[str, Any], sample_index: int) -> list[dict[str, str]]:
    source_messages = sample.get("messages")
    if not isinstance(source_messages, list):
        raise ValueError(f"Sample {sample_index} does not contain a messages list.")

    messages = []
    for message in source_messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role in {"system", "user"} and isinstance(content, str) and content:
            messages.append({"role": role, "content": content})

    roles = [message["role"] for message in messages]
    if roles != ["system", "user"]:
        raise ValueError(
            f"Sample {sample_index} must contain system then user; found {roles}."
        )
    messages[0]["content"] += FORMAT_ENFORCEMENT
    return messages


def extract_medical_text(user_content: str) -> str:
    marker = "Medical text:\n"
    if marker not in user_content:
        raise ValueError("User message is missing the 'Medical text:' marker.")
    medical_text = user_content.rsplit(marker, 1)[1].strip()
    if not medical_text:
        raise ValueError("Medical text is empty.")
    return medical_text


def response_message_text(message: Any) -> tuple[str, str]:
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        content = "" if content is None else str(content)
    reasoning = getattr(message, "reasoning_content", None)
    if not isinstance(reasoning, str):
        reasoning = ""
    return content.strip(), reasoning.strip()


def parse_model_response(
    content: str, reasoning_content: str
) -> tuple[str, dict[str, Any], dict[str, bool]]:
    think_match = re.search(
        r"<think>(.*?)</think>", content, flags=re.DOTALL | re.IGNORECASE
    )
    output_match = re.search(
        r"<output>(.*?)</output>", content, flags=re.DOTALL | re.IGNORECASE
    )

    cot = think_match.group(1).strip() if think_match else reasoning_content.strip()
    if not cot:
        raise ValueError("The response contains no usable COT text.")

    output_text = output_match.group(1).strip() if output_match else content
    output_text = output_text.replace("```json", "").replace("```", "").strip()
    start = output_text.find("{")
    end = output_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object was found in the response.")

    output = json.loads(output_text[start : end + 1])
    if not isinstance(output, dict):
        raise ValueError("The parsed output must be a JSON object.")

    flags = {
        "response_had_think_tag": think_match is not None,
        "response_had_output_tag": output_match is not None,
        "response_had_reasoning_content": bool(reasoning_content),
    }
    return cot, output, flags


def unique_entity_id(
    requested_id: str,
    label: str,
    counters: dict[str, int],
    used_ids: set[str],
) -> str:
    if requested_id and requested_id not in used_ids:
        return requested_id

    prefix = LABEL_ID_PREFIX[label]
    while True:
        counters[prefix] = counters.get(prefix, 0) + 1
        candidate = f"{prefix}{counters[prefix]}"
        if candidate not in used_ids:
            return candidate


def normalize_output(output: dict[str, Any]) -> dict[str, Any]:
    raw_entities = output.get("entities")
    if not isinstance(raw_entities, list):
        raw_entities = output.get("nodes")
    raw_relations = output.get("relations")
    if not isinstance(raw_relations, list):
        raw_relations = output.get("edges")
    if not isinstance(raw_entities, list) or not raw_entities:
        raise ValueError("Output must contain a non-empty entities list.")
    if not isinstance(raw_relations, list):
        raise ValueError("Output must contain a relations list.")

    entities = []
    aliases: dict[str, str] = {}
    used_ids: set[str] = set()
    counters: dict[str, int] = {}

    for position, entity in enumerate(raw_entities):
        if not isinstance(entity, dict):
            raise ValueError(f"Entity {position} is not an object.")

        raw_label = str(entity.get("label") or "").strip()
        label = LABEL_NORMALIZATION.get(raw_label, raw_label)
        if label not in ALLOWED_LABELS:
            raise ValueError(f"Entity {position} has unsupported label {raw_label!r}.")

        old_id = str(entity.get("id") or "").strip()
        node_name = str(
            entity.get("Node Name") or entity.get("node_name") or ""
        ).strip()
        if node_name in GENERIC_NODE_NAMES:
            node_name = ""
        name = str(
            entity.get("name") or entity.get("text") or node_name or old_id
        ).strip()
        if not name:
            raise ValueError(f"Entity {position} has neither name nor id.")

        entity_id = unique_entity_id(old_id, label, counters, used_ids)
        used_ids.add(entity_id)
        properties = entity.get("properties")
        if not isinstance(properties, dict):
            properties = {}

        entities.append(
            {
                "id": entity_id,
                "label": label,
                "name": name,
                "properties": properties,
            }
        )
        for alias in {old_id, name, entity_id}:
            if alias:
                aliases[alias] = entity_id

    relations = []
    seen_relations: set[tuple[str, str, str]] = set()
    for position, relation in enumerate(raw_relations):
        if not isinstance(relation, dict):
            raise ValueError(f"Relation {position} is not an object.")

        source_raw = str(relation.get("source") or "").strip()
        target_raw = str(relation.get("target") or "").strip()
        source = aliases.get(source_raw)
        target = aliases.get(target_raw)
        if not source or not target:
            raise ValueError(
                f"Relation {position} has an unresolved endpoint: "
                f"{source_raw!r} -> {target_raw!r}."
            )

        relation_slug = ""
        for candidate in [
            relation.get("relation"),
            relation.get("Relation"),
            relation.get("type"),
            relation.get("name"),
            relation.get("relation_name"),
            relation.get("Relation Name"),
        ]:
            candidate_text = str(candidate or "").strip()
            if candidate_text in RELATION_CONTRACT:
                relation_slug = candidate_text
                break
        contract = RELATION_CONTRACT.get(relation_slug)
        if contract is None:
            raise ValueError(
                f"Relation {position} has unsupported relation {relation_slug!r}."
            )
        relation_type, relation_name = contract

        relation_key = (source, target, relation_slug)
        if relation_key in seen_relations:
            continue
        seen_relations.add(relation_key)
        evidence = relation.get("evidence")
        if not isinstance(evidence, str):
            relation_properties = relation.get("properties")
            if isinstance(relation_properties, dict):
                evidence = relation_properties.get("description")
        if not isinstance(evidence, str):
            evidence = ""

        relations.append(
            {
                "source": source,
                "target": target,
                "relation_type": relation_type,
                "relation_name": relation_name,
                "relation": relation_slug,
                "evidence": evidence,
            }
        )

    return {"entities": entities, "relations": relations}


def build_assistant_content(cot: str, output: dict[str, Any]) -> str:
    output_json = json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)
    return f"<think>\n{cot.strip()}\n</think>\n<output>\n{output_json}\n</output>"


def usage_dict(completion: Any) -> dict[str, Any]:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return {}
    result = {}
    for field in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        value = getattr(usage, field, None)
        if value is not None:
            result[field] = value
    return result


def load_existing_records(overwrite: bool) -> list[dict[str, Any]]:
    if overwrite or not generated_output_path.exists():
        return []
    existing = load_json(generated_output_path)
    if not isinstance(existing, list):
        raise ValueError(f"Existing output is not a list: {generated_output_path}")
    return [item for item in existing if isinstance(item, dict)]


def build_generated_record(
    index: int,
    manifest: dict[str, Any],
    medical_text: str,
    cot: str,
    output: dict[str, Any],
    flags: dict[str, bool],
) -> dict[str, Any]:
    source = str(manifest.get("source") or "")
    source_index = manifest.get("source_index")
    return {
        "input": medical_text,
        "input_used": medical_text,
        "cot": cot,
        "output": output,
        "success": True,
        "global_idx": index,
        "source_id": f"{source}:{source_index}",
        "source": source,
        "source_index": source_index,
        "code": "",
        "title": manifest.get("title", ""),
        "input_chars": len(medical_text),
        "original_input_chars": len(medical_text),
        "cot_style": "template",
        "model": model_name,
        **flags,
    }


def recover_paid_failures(
    source_dataset: list[dict[str, Any]],
    manifest_records: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if not metadata_output_path.exists():
        raise ValueError(f"Metadata file does not exist: {metadata_output_path}")
    metadata = load_json(metadata_output_path)
    paid_failures = metadata.get("failures_detail")
    if not isinstance(paid_failures, list):
        raise ValueError("Metadata does not contain a failures_detail list.")

    diagnostics = metadata.get("generation_diagnostics")
    if not isinstance(diagnostics, list):
        diagnostics = []
    remaining_failures = []
    recovered_indices = []

    for failure in paid_failures:
        index = failure.get("global_idx") if isinstance(failure, dict) else None
        if not isinstance(index, int) or not 0 <= index < len(source_dataset):
            remaining_failures.append(failure)
            continue
        raw_response = str(failure.get("raw_response") or "")
        reasoning_content = str(failure.get("reasoning_content") or "")
        try:
            messages = request_messages(source_dataset[index], index)
            medical_text = extract_medical_text(messages[1]["content"])
            cot, raw_output, flags = parse_model_response(
                raw_response, reasoning_content
            )
            output = normalize_output(raw_output)
            record = build_generated_record(
                index,
                manifest_records[index],
                medical_text,
                cot,
                output,
                flags,
            )
            records = [item for item in records if item.get("global_idx") != index]
            records.append(record)
            diagnostics.append(
                {
                    "global_idx": index,
                    "recovered_from_paid_raw_response": True,
                    "entity_count": len(output["entities"]),
                    "relation_count": len(output["relations"]),
                }
            )
            recovered_indices.append(index)
        except Exception as exc:
            updated_failure = dict(failure)
            updated_failure["local_recovery_error_type"] = type(exc).__name__
            updated_failure["local_recovery_error"] = str(exc)
            remaining_failures.append(updated_failure)

    print(f"Locally recovered paid responses: {recovered_indices}")
    print(
        "Still require a new request: "
        f"{[item.get('global_idx') for item in remaining_failures]}"
    )
    return records, remaining_failures, diagnostics


def checkpoint(
    source_dataset: list[dict[str, Any]],
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    selected_indices: list[int],
) -> None:
    records.sort(key=lambda item: item.get("global_idx", 10**12))
    write_json(generated_output_path, records)

    completed_dataset = json.loads(json.dumps(source_dataset, ensure_ascii=False))
    record_by_index = {item.get("global_idx"): item for item in records}
    for index, sample in enumerate(completed_dataset):
        record = record_by_index.get(index)
        if record is None:
            continue
        sample["messages"][-1]["content"] = build_assistant_content(
            record["cot"], record["output"]
        )
    write_json(completed_output_path, completed_dataset)

    completed_selected = sum(index in record_by_index for index in selected_indices)
    metadata = {
        "input": str(dataset_path),
        "manifest": str(manifest_path),
        "model": model_name,
        "api_base": api_base,
        "automatic_retries": 0,
        "selected_indices": selected_indices,
        "selected_samples": len(selected_indices),
        "completed_selected_samples": completed_selected,
        "records": len(records),
        "failures": len(failures),
        "generation_diagnostics": diagnostics,
        "failures_detail": failures,
        "generated_output": str(generated_output_path),
        "completed_llamafactory_output": str(completed_output_path),
    }
    write_json(metadata_output_path, metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the 22 treatment-related samples with Gemini."
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First zero-based source index to process (default: 0).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Optional number of samples to process; default processes the remainder.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to wait after each paid request (default: 1).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard checkpoints and request selected samples again.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Validate inputs and show the plan without sending API requests.",
    )
    parser.add_argument(
        "--recover-only",
        action="store_true",
        help=(
            "Reprocess raw responses already saved in metadata, write recovered "
            "checkpoints, and exit without any API request."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dataset, manifest_records = load_source_data()

    if not 0 <= args.start_index < len(source_dataset):
        raise SystemExit(
            f"--start-index must be between 0 and {len(source_dataset) - 1}."
        )
    end_index = len(source_dataset)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise SystemExit("--max-samples must be greater than zero.")
        end_index = min(end_index, args.start_index + args.max_samples)
    selected_indices = list(range(args.start_index, end_index))

    existing_records = load_existing_records(args.overwrite)

    if args.recover_only:
        if args.overwrite:
            raise SystemExit("--recover-only cannot be combined with --overwrite.")
        records, failures, diagnostics = recover_paid_failures(
            source_dataset,
            manifest_records,
            existing_records,
        )
        checkpoint(
            source_dataset,
            records,
            failures,
            diagnostics,
            selected_indices,
        )
        record_indices = {
            item.get("global_idx")
            for item in records
            if item.get("success") is True
        }
        missing = [
            index for index in selected_indices if index not in record_indices
        ]
        print("Recovery-only mode completed; no API request was sent.")
        print(f"Completed selected samples: {len(selected_indices) - len(missing)}")
        print(f"Missing indices: {missing}")
        return

    completed_indices = {
        item.get("global_idx")
        for item in existing_records
        if item.get("success") is True
    }
    pending_indices = [
        index for index in selected_indices if index not in completed_indices
    ]

    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_name}")
    print(f"Endpoint: {api_base}/chat/completions")
    print(f"Selected samples: {len(selected_indices)}")
    print(f"Already completed: {len(selected_indices) - len(pending_indices)}")
    print(f"Pending paid requests: {len(pending_indices)}")
    print("Automatic retries: 0")
    print(f"Generated output: {generated_output_path}")
    print(f"Completed dataset: {completed_output_path}")

    if args.preview:
        print("Preview completed; no API request was sent.")
        return

    resolved_api_key = os.getenv("XI_AI_API_KEY", "").strip() or api_key.strip()
    if not resolved_api_key:
        raise SystemExit(
            "API key is empty. Fill api_key near the top of this script or set "
            "the XI_AI_API_KEY environment variable."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'openai'. Install it with: "
            "python -m pip install openai"
        ) from exc

    client = OpenAI(
        api_key=resolved_api_key,
        base_url=api_base,
        timeout=600.0,
    )
    records = existing_records
    failures: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for progress, index in enumerate(pending_indices, start=1):
        sample = source_dataset[index]
        manifest = manifest_records[index]
        messages = request_messages(sample, index)
        medical_text = extract_medical_text(messages[1]["content"])
        print(
            f"\n[{progress}/{len(pending_indices)}] index={index} "
            f"source={manifest.get('source')} chars={len(medical_text)}"
        )

        response_content = ""
        reasoning_content = ""
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            if not completion.choices:
                raise ValueError("The API response contains no choices.")

            choice = completion.choices[0]
            response_content, reasoning_content = response_message_text(choice.message)
            cot, raw_output, flags = parse_model_response(
                response_content, reasoning_content
            )
            output = normalize_output(raw_output)

            record = build_generated_record(
                index,
                manifest,
                medical_text,
                cot,
                output,
                flags,
            )
            records = [item for item in records if item.get("global_idx") != index]
            records.append(record)
            diagnostics.append(
                {
                    "global_idx": index,
                    "response_id": getattr(completion, "id", None),
                    "response_model": getattr(completion, "model", None),
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "entity_count": len(output["entities"]),
                    "relation_count": len(output["relations"]),
                    **usage_dict(completion),
                }
            )
            print(
                f"Success: entities={len(output['entities'])}, "
                f"relations={len(output['relations'])}"
            )
        except Exception as exc:
            failures.append(
                {
                    "global_idx": index,
                    "source": manifest.get("source"),
                    "source_index": manifest.get("source_index"),
                    "title": manifest.get("title", ""),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "raw_response": response_content,
                    "reasoning_content": reasoning_content,
                }
            )
            print(f"Failed without retry: {type(exc).__name__}: {exc}")

        checkpoint(
            source_dataset,
            records,
            failures,
            diagnostics,
            selected_indices,
        )
        if args.sleep and progress < len(pending_indices):
            time.sleep(args.sleep)

    record_indices = {
        item.get("global_idx") for item in records if item.get("success") is True
    }
    missing = [index for index in selected_indices if index not in record_indices]
    print("\nGeneration finished.")
    print(f"Completed selected samples: {len(selected_indices) - len(missing)}")
    print(f"Failed/missing selected samples: {len(missing)}")
    if missing:
        print(f"Missing indices: {missing}")
    print(f"Generated output: {generated_output_path}")
    print(f"Completed dataset: {completed_output_path}")
    print(f"Metadata: {metadata_output_path}")


if __name__ == "__main__":
    main()
