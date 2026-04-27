#!/usr/bin/env python3
import argparse
import json
import gc
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_csv_list(text: str | None):
    if not text:
        return None
    items = [part.strip() for part in text.split(",")]
    return [item for item in items if item]


def resolve_records(data):
    return data["entities"] if isinstance(data, dict) and "entities" in data else data


def extract_text_from_record(record):
    fields = ["title", "definition", "longDefinition", "diagnostic_criteria", "diagnosticCriteria", "description"]
    return "\n".join(str(record.get(k)) for k in fields if record.get(k))


def truncate_at_boundary(text: str, max_chars: int):
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    window = text[:max_chars]
    boundary_candidates = ["\n\n", "\n### ", "\n## ", "\n- ", "\n", ". ", ".\n", "。", "；", "; "]
    best_cut = -1
    for marker in boundary_candidates:
        pos = window.rfind(marker)
        if pos > best_cut:
            best_cut = pos + len(marker.rstrip())
    if best_cut >= int(max_chars * 0.7):
        return window[:best_cut].rstrip()
    return window.rstrip()


def build_user_prompt(text: str):
    return (
        "Perform a high-coverage extraction over the following medical text and output a structured JSON result.\n\n"
        f"Medical text:\n{text}\n\n"
        "Steps:\n"
        "1) Mark key entities.\n"
        "2) Extract entities and properties using only explicitly stated information.\n"
        "3) Build only well-supported relations.\n"
        "4) Verify that every relation endpoint exists in `entities`.\n"
        "5) Check whether you missed explicitly listed symptoms, diagnostic criteria, differential diagnoses, subtypes, risk factors, or patient features.\n\n"
        f"{HIGH_COVERAGE_REQUIREMENTS}\n\n"
        "Output the JSON object directly, with no markdown code blocks."
    )


def load_model_bundle(base_model: str, adapter_path: str | None, load_in_8bit: bool = False, load_in_4bit: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def build_base_model():
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        elif load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        else:
            model_kwargs["dtype"] = torch.bfloat16

        return AutoModelForCausalLM.from_pretrained(
            base_model,
            **model_kwargs,
        )

    model = build_base_model()
    if adapter_path:
        try:
            model = PeftModel.from_pretrained(model, adapter_path)
        except AttributeError as exc:
            # Some PEFT configs reference float8 dtypes that are not exposed by older torch builds.
            # Retrying with adapter dtype autocast disabled usually avoids the incompatible code path.
            if "float8_e8m0fnu" not in str(exc):
                raise
            print(
                f"Warning: torch build does not expose float8_e8m0fnu; "
                f"retrying adapter load without dtype autocast for {adapter_path}"
            )
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = build_base_model()
            model = PeftModel.from_pretrained(
                model,
                adapter_path,
                autocast_adapter_dtype=False,
            )
    model.eval()
    return tokenizer, model


def generate_once(tokenizer, model, user_prompt: str, max_new_tokens: int):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Compare base Qwen, specific adapter, and standard adapter outputs.")
    parser.add_argument("--data", help="Path to mental_disorders json. Optional when --samples already includes source_record.")
    parser.add_argument("--samples", required=True, help="Path to sample cases json.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B", help="Base model path or repo id.")
    parser.add_argument("--specific-adapter", required=True, help="Specific adapter directory.")
    parser.add_argument("--standard-adapter", required=True, help="Standard adapter directory.")
    parser.add_argument("--output", required=True, help="Output directory or output json path prefix.")
    parser.add_argument("--max-input-chars", type=int, default=12000)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--models",
        default="base_qwen,specific_adapter,standard_adapter",
        help="Comma-separated subset of models to run: base_qwen,specific_adapter,standard_adapter",
    )
    parser.add_argument(
        "--case-ids",
        help="Comma-separated subset of sample ids to run, e.g. 18,149",
    )
    parser.add_argument("--load-in-8bit", action="store_true", help="Load base model in 8-bit to reduce memory usage.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load base model in 4-bit to reduce memory usage further.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.load_in_8bit and args.load_in_4bit:
        raise SystemExit("Use at most one of --load-in-8bit or --load-in-4bit.")

    output_arg = Path(args.output)
    sample_cases = load_json(Path(args.samples))
    if not isinstance(sample_cases, list):
        raise SystemExit("--samples must be a JSON list.")

    selected_case_ids = parse_csv_list(args.case_ids)
    if selected_case_ids is not None:
        selected_case_ids = {int(item) for item in selected_case_ids}
        sample_cases = [case for case in sample_cases if case.get("id") in selected_case_ids]
        if not sample_cases:
            raise SystemExit("No sample cases matched --case-ids.")

    records = None
    if args.data:
        data = load_json(Path(args.data))
        records = resolve_records(data)
        if not isinstance(records, list):
            raise SystemExit("--data must resolve to a JSON list or a dict with an 'entities' list.")

    results = []
    for case in sample_cases:
        idx = case["id"]
        record = case.get("source_record")
        if not isinstance(record, dict):
            if records is None:
                raise SystemExit("Each sample must include source_record when --data is not provided.")
            record = records[idx]
        full_text = extract_text_from_record(record)
        used_text = truncate_at_boundary(full_text, args.max_input_chars)
        results.append(
            {
                "id": idx,
                "title": case.get("title"),
                "label": case.get("label"),
                "input_chars": len(used_text),
                "input_used": used_text,
                "outputs": {},
            }
        )

    model_specs = {
        "base_qwen": None,
        "specific_adapter": args.specific_adapter,
        "standard_adapter": args.standard_adapter,
    }
    selected_models = parse_csv_list(args.models) or []
    invalid_models = [name for name in selected_models if name not in model_specs]
    if invalid_models:
        raise SystemExit(f"Unsupported model names in --models: {', '.join(invalid_models)}")

    if output_arg.suffix.lower() == ".json":
        output_dir = output_arg.parent
        output_prefix = output_arg.stem
    else:
        output_dir = output_arg
        output_prefix = "qwen_compare"

    for name in selected_models:
        adapter_path = model_specs[name]
        print(f"Loading {name}...")
        tokenizer, model = load_model_bundle(
            args.base_model,
            adapter_path,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
        )
        model_results = []
        try:
            for item in results:
                user_prompt = build_user_prompt(item["input_used"])
                output_text = generate_once(tokenizer, model, user_prompt, args.max_new_tokens)
                model_item = {
                    "id": item["id"],
                    "title": item["title"],
                    "label": item["label"],
                    "input_chars": item["input_chars"],
                    "input_used": item["input_used"],
                    "model": name,
                    "output": output_text,
                }
                model_results.append(model_item)
        finally:
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        output_path = output_dir / f"{output_prefix}_{name}.json"
        dump_json(output_path, model_results)
        print(f"Saved {name} results to {output_path}")

    print(f"Saved per-model outputs under {output_dir}")


if __name__ == "__main__":
    main()
