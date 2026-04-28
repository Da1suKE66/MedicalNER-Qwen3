#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = """
You are an expert in constructing medical knowledge graphs. Extract entities and relations from medical text and output a JSON.

# Node information

| label | Node Name | properties |
|-------|-----------|------------|
| Disease | Disease (DSM-5) | DSM-5 Code, Subtype, Course Requirements, Comorbidity Types, Prognosis Factors |
| Symptom | Core Symptom | Symptom Description, Occurrence Frequency, Severity Description |
| Symptom | Associated Symptom | Symptom Description, Occurrence Frequency, Severity Description, Diagnostic Value |
| Disease | Differential Diagnosis | DSM-5 Code, Core Features, Key Differentiation Points, Misdiagnosis Risk |
| Diagnostic Criteria | DSM-5 Diagnostic Criteria | Required Core Symptoms, Functional Impairment Requirements, Exclusion Details |
| Interview Tool | Key Interview Points | Key Inquiry Directions, Exclusions, Sample Interview Phrases, Follow-up Focus |
| Patient Information | Patient Features | Age Group, Comorbidities, Special Conditions, Medication History |
| Medication | Drug | Generic Name, Indications, Contraindications, Dosage for Special Populations, Common Side Effects |
| Communication Method | Dialogue Strategy | Suitable Patient Type, Empathetic Phrases, Pitfalls to Avoid |
| Risk Information | Risk Factors | Risk Type, Alert Keywords, Emergency Intervention Steps |

# Relation information

| Relation Type | Relation Name | Relation |
|---------------|---------------|---------|
| Disease Hierarchy | Subsumes | subsumes |
| Disease Hierarchy | Differentiates From | differentiates_from |
| Disease Hierarchy | Co-occurrence Frequency | co_occurs_with_frequency |
| Disease Hierarchy | Associated with Poor Prognosis | associated_with_poor_prognosis_in |
| Symptom–Disease | Core Symptom Of | is_core_symptom_of |
| Symptom–Disease | Associated Symptom Of | is_associated_symptom_of |
| Symptom–Disease | Precedes | precedes |
| Symptom–Disease | Follows | follows |
| Symptom–Disease | Modulated By | modulated_by |
| Diagnostic Criteria–Disease–Symptom Triangle | Required For Diagnosis Of | required_for_diagnosis_of |
| Diagnostic Criteria–Disease–Symptom Triangle | Excludes If Present | excludes_if_present |
| Diagnostic Criteria–Disease–Symptom Triangle | Supports Subtyping Of | supports_subtyping_of |
| Interview Tool–Symptom–Patient Info | Assesses For | assesses_for |
| Interview Tool–Symptom–Patient Info | Triggers Follow-up Question On | triggers_follow_up_question_on |
| Interview Tool–Symptom–Patient Info | Informed By Patient Demographics | informed_by_patient_demographics |
| Medication–Disease–Patient Info | First Line For | first_line_for |
| Medication–Disease–Patient Info | Contraindicated In | contraindicated_in |
| Medication–Disease–Patient Info | Dose Adjusted For | dose_adjusted_for |
| Medication–Disease–Patient Info | Interacts With | interacts_with |
| Communication–Patient–Risk | Recommended For | recommended_for |
| Communication–Patient–Risk | Avoid With | avoid_with |
| Communication–Patient–Risk | Escalates To | escalates_to |
| Risk–Disease–Symptom | Triggers Alert When | triggers_alert_when |
| Risk–Disease–Symptom | Mediated By | mediated_by |

Extract only explicitly mentioned information in the text; do not infer.
If a property is missing, leave it empty.
The relation's source and target must exist in the entities.
IMPORTANT: Output strict JSON only. Do not include markdown code blocks or any extra text.
""".strip()

HIGH_COVERAGE_REQUIREMENTS = """Coverage requirements:
1) Prefer high coverage over short summarization.
2) Keep explicitly listed symptoms, criteria, subtypes, differential diagnoses, patient features, and risk factors whenever they appear.
3) Never infer; keep only text-supported information."""

QWEN_THINK_OPEN = "<think>"
QWEN_THINK_CLOSE = "</think>"


def normalize_qwen_think_tags(text: str) -> str:
    return (
        str(text)
        .replace("<thinking>", QWEN_THINK_OPEN)
        .replace("</thinking>", QWEN_THINK_CLOSE)
    )


COT_TEMPLATE_PROMPT = """<think>
Write a concise reasoning trace in no more than 6 steps, and keep the total length as short as possible.
</think>
<output>
Write the final JSON object here.
</output>

Requirements:
1) Extract only explicitly stated information.
2) `<output>` must be strict JSON.
3) Prioritize a complete `<output>` over a longer thinking trace."""

COT_SPECIFIC_PROMPT = """<think>
Write only extraction decisions that are specific to this sample, and do not repeat a generic workflow template.
Focus on:
1) the most important disease, subtype, or severity information in this text
2) the most important symptoms, diagnostic criteria, differential diagnoses, patient features, or risk information worth preserving
3) which relations are the most important and why they should be kept
4) which seemingly relevant items were not extracted because the text was not explicit enough

Constraints:
- Do not write generic steps such as "identify the disease, then extract symptoms, then build relations"
- Do not restate the whole text
- Keep it within 4 bullets or short items, and keep the total length as short as possible
</think>
<output>
Write the final JSON object here.
</output>

Requirements:
1) Extract only explicitly stated information.
2) `<output>` must be strict JSON.
3) Prioritize a complete `<output>` over a longer thinking trace."""


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def normalize_role(role: str) -> str:
    mapping = {
        "system": "system",
        "user": "user",
        "assistant": "assistant",
        "human": "user",
        "gpt": "assistant",
    }
    return mapping.get(str(role).strip().lower(), str(role).strip().lower())


def infer_cot_style(item: dict, input_path: Path, fallback: str) -> str:
    style = item.get("cot_style")
    if style in {"template", "specific"}:
        return style

    name = input_path.name.lower()
    if "specific" in name:
        return "specific"
    if "standard" in name or "template" in name:
        return "template"
    return fallback


def build_generation_user_prompt(model_input: str, cot_enabled: bool, cot_style: str) -> str:
    if cot_enabled:
        cot_instruction = COT_SPECIFIC_PROMPT if cot_style == "specific" else COT_TEMPLATE_PROMPT
        return (
            "Read the following medical text and strictly follow this output format:\n\n"
            f"{cot_instruction}\n"
            f"{HIGH_COVERAGE_REQUIREMENTS}\n\n"
            f"Medical text:\n{model_input}"
        )

    return (
        "Perform a high-coverage extraction over the following medical text and output a structured JSON result.\n\n"
        f"Medical text:\n{model_input}\n\n"
        "Steps:\n"
        "1) Mark key entities.\n"
        "2) Extract entities and properties using only explicitly stated information.\n"
        "3) Build only well-supported relations.\n"
        "4) Verify that every relation endpoint exists in `entities`.\n"
        "5) Check whether you missed explicitly listed symptoms, diagnostic criteria, differential diagnoses, subtypes, risk factors, or patient features.\n\n"
        f"{HIGH_COVERAGE_REQUIREMENTS}\n\n"
        "Output the JSON object directly, with no markdown code blocks."
    )


def build_assistant_text(output: dict, cot_text: str):
    final_json = json.dumps(output, ensure_ascii=False, indent=2)
    cot_text = (cot_text or "").strip()
    if cot_text:
        return (
            f"{QWEN_THINK_OPEN}\n"
            f"{cot_text}\n"
            f"{QWEN_THINK_CLOSE}\n"
            "<output>\n"
            f"{final_json}\n"
            "</output>"
        )
    return final_json


def convert_conversations_item(item: dict):
    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        return None

    messages = []
    for msg in conversations:
        if not isinstance(msg, dict):
            continue
        role = normalize_role(msg.get("from", ""))
        content = msg.get("value")
        if role not in {"system", "user", "assistant"}:
            continue
        if content is None:
            continue
        messages.append({"role": role, "content": normalize_qwen_think_tags(content)})

    if len(messages) < 2:
        return None
    return {"messages": messages}


def convert_cot_item(item: dict, input_path: Path, fallback_cot_style: str):
    if not isinstance(item, dict):
        return None
    if item.get("success") is False:
        return None
    if "input" not in item or "output" not in item:
        return None

    model_input = item.get("input_used") or item["input"]
    if isinstance(model_input, list):
        model_input = "\n\n".join(str(part) for part in model_input if part)
    cot_text = str(item.get("cot") or "").strip()
    cot_enabled = bool(cot_text)
    cot_style = infer_cot_style(item, input_path, fallback_cot_style)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_generation_user_prompt(model_input, cot_enabled, cot_style)},
            {"role": "assistant", "content": build_assistant_text(item["output"], cot_text)},
        ]
    }


def convert_chunk_traces(item: dict, input_path: Path, fallback_cot_style: str):
    traces = item.get("chunk_traces")
    if not isinstance(traces, list):
        return []

    converted = []
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        if trace.get("success") is False:
            continue
        if "output" not in trace or "input_used" not in trace:
            continue

        cot_text = str(trace.get("cot") or "").strip()
        cot_enabled = bool(cot_text)
        cot_style = trace.get("cot_style") or infer_cot_style(item, input_path, fallback_cot_style)

        converted.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_generation_user_prompt(str(trace["input_used"]), cot_enabled, cot_style)},
                    {"role": "assistant", "content": build_assistant_text(trace["output"], cot_text)},
                ]
            }
        )
    return converted


def convert_records(records, input_path: Path, fallback_cot_style: str, include_chunk_traces: bool = False, chunk_traces_only: bool = False):
    converted = []
    skipped = 0

    for item in records:
        converted_items = []
        if isinstance(item, dict) and "conversations" in item:
            converted_item = convert_conversations_item(item)
            if converted_item is not None:
                converted_items.append(converted_item)
        elif isinstance(item, dict):
            if not chunk_traces_only:
                converted_item = convert_cot_item(item, input_path, fallback_cot_style)
                if converted_item is not None:
                    converted_items.append(converted_item)
            if include_chunk_traces:
                converted_items.extend(convert_chunk_traces(item, input_path, fallback_cot_style))

        if not converted_items:
            skipped += 1
            continue
        converted.extend(converted_items)

    return converted, skipped


def update_dataset_info(dataset_info_path: Path, dataset_name: str, output_file: Path):
    if dataset_info_path.exists():
        payload = load_json(dataset_info_path)
    else:
        payload = {}

    payload[dataset_name] = {
        "file_name": output_file.name,
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }
    dump_json(dataset_info_path, payload)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert KG extraction data to LLaMA-Factory dataset format.")
    parser.add_argument("--input", required=True, help="Input JSON path.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--dataset-info", required=True, help="dataset_info.json path to create or update.")
    parser.add_argument("--dataset-name", default="kg_ner_sft", help="Dataset name registered in dataset_info.json.")
    parser.add_argument("--fallback-cot-style", choices=["template", "specific"], default="template", help="Fallback cot_style when the source json does not record it.")
    parser.add_argument("--include-chunk-traces", action="store_true", help="Also export successful chunk_traces as standalone SFT samples.")
    parser.add_argument("--chunk-traces-only", action="store_true", help="Only export successful chunk_traces, skip merged top-level records.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    dataset_info_path = Path(args.dataset_info)

    records = load_json(input_path)
    if not isinstance(records, list):
        raise SystemExit("Input JSON must be a list.")

    converted, skipped = convert_records(
        records,
        input_path=input_path,
        fallback_cot_style=args.fallback_cot_style,
        include_chunk_traces=args.include_chunk_traces,
        chunk_traces_only=args.chunk_traces_only,
    )
    if not converted:
        raise SystemExit("No valid samples found after conversion.")

    dump_json(output_path, converted)
    update_dataset_info(dataset_info_path, args.dataset_name, output_path)

    print(f"Converted samples: {len(converted)}")
    print(f"Skipped samples: {skipped}")
    print(f"Output dataset: {output_path}")
    print(f"Updated dataset info: {dataset_info_path}")


if __name__ == "__main__":
    main()
