"""
Run Gemini on split ICD inputs produced by split_gemini_generation.py.

This script is intentionally separate from generate_ner_dataset_gemini.py because
it writes merge-ready records instead of conversation-style training samples.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm


KG_SCHEMA_PROMPT = """
You are an expert in constructing medical knowledge graphs. Extract entities and relations from medical text and output strict JSON.

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
| Symptom-Disease | Core Symptom Of | is_core_symptom_of |
| Symptom-Disease | Associated Symptom Of | is_associated_symptom_of |
| Symptom-Disease | Precedes | precedes |
| Symptom-Disease | Follows | follows |
| Symptom-Disease | Modulated By | modulated_by |
| Diagnostic Criteria-Disease-Symptom Triangle | Required For Diagnosis Of | required_for_diagnosis_of |
| Diagnostic Criteria-Disease-Symptom Triangle | Excludes If Present | excludes_if_present |
| Diagnostic Criteria-Disease-Symptom Triangle | Supports Subtyping Of | supports_subtyping_of |
| Interview Tool-Symptom-Patient Info | Assesses For | assesses_for |
| Interview Tool-Symptom-Patient Info | Triggers Follow-up Question On | triggers_follow_up_question_on |
| Interview Tool-Symptom-Patient Info | Informed By Patient Demographics | informed_by_patient_demographics |
| Medication-Disease-Patient Info | First Line For | first_line_for |
| Medication-Disease-Patient Info | Contraindicated In | contraindicated_in |
| Medication-Disease-Patient Info | Dose Adjusted For | dose_adjusted_for |
| Medication-Disease-Patient Info | Interacts With | interacts_with |
| Communication-Patient-Risk | Recommended For | recommended_for |
| Communication-Patient-Risk | Avoid With | avoid_with |
| Communication-Patient-Risk | Escalates To | escalates_to |
| Risk-Disease-Symptom | Triggers Alert When | triggers_alert_when |
| Risk-Disease-Symptom | Mediated By | mediated_by |

Rules:
- Use only the labels, properties, relation types, relation names, and relations listed above.
- Entity label must be exactly one of: Disease, Symptom, Diagnostic Criteria, Interview Tool, Patient Information, Medication, Communication Method, Risk Information.
- Do not use Node Name values such as Core Symptom, Associated Symptom, or Differential Diagnosis as labels.
- Extract only explicitly stated information; do not infer. If a property is missing, leave it empty.
- For ICD-11 inputs, put the source code value in the "DSM-5 Code" property when a code is present.
- Use English source wording only. Do not translate terms into Chinese or rewrite medical names.
- The relation source and target must be entity ids from the entities list.
- Output strict JSON only, with this shape:
{"source_id":"","code":"","title":"","entities":[{"id":"D1","label":"Disease","name":"","properties":{}}],"relations":[{"source":"S1","target":"D1","relation_type":"Symptom-Disease","relation_name":"Core Symptom Of","relation":"is_core_symptom_of","evidence":""}]}
""".strip()


STRUCTURE_SYSTEM_PROMPT = (
    KG_SCHEMA_PROMPT
    + "\n\nTask for this input: use ICD structural fields only. Prefer parentResolved, childResolved, ancestorResolved, and descendantResolved over raw URL lists when title/code is available. Extract disease identity, explicit hierarchy/subtype facts, exclusions, inclusions, index terms, synonyms, coding notes, and postcoordination information only when they fit the schema above. Do not extract symptoms, diagnostic criteria, or unsupported properties in structure mode."
)


LONG_TEXT_SYSTEM_PROMPT = (
    KG_SCHEMA_PROMPT
    + "\n\nTask for this input: use definition, diagnosticCriteria, and narrowerTerms only. Extract diagnostic criteria, symptoms, differential diagnoses, patient features, risk/prognosis factors, and explicitly stated medication or communication facts only when present."
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in Gemini response")
    return json.loads(cleaned[start : end + 1])


def safe_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    name = getattr(value, "name", None)
    if name:
        return name
    return str(value)


def response_diagnostics(response: Any, prompt: str) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {"prompt_chars": len(prompt)}

    candidates = getattr(response, "candidates", None) or []
    if candidates:
        candidate = candidates[0]
        diagnostics["finish_reason"] = safe_scalar(getattr(candidate, "finish_reason", None))

    usage = getattr(response, "usage_metadata", None)
    if usage:
        for attr in [
            "prompt_token_count",
            "candidates_token_count",
            "total_token_count",
            "thoughts_token_count",
            "cached_content_token_count",
        ]:
            value = getattr(usage, attr, None)
            if value is not None:
                diagnostics[attr] = safe_scalar(value)

    return diagnostics


def build_prompt(mode: str, entity: dict[str, Any]) -> str:
    if mode == "structure":
        system_prompt = STRUCTURE_SYSTEM_PROMPT
    elif mode == "long_text":
        system_prompt = LONG_TEXT_SYSTEM_PROMPT
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return (
        f"{system_prompt}\n\n"
        "Input JSON:\n"
        f"{json.dumps(entity, ensure_ascii=False, indent=2)}\n\n"
        "Output strict JSON only:"
    )


def run_generation(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    load_dotenv(script_dir / ".env")
    load_dotenv()

    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY. Set it in .env or pass --api_key.")

    data = load_json(args.input)
    entities = data.get("entities", [])
    if args.offset:
        entities = entities[args.offset :]
    if args.max_samples is not None:
        entities = entities[: args.max_samples]

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(args.timeout * 1000)),
    )

    records = []
    failures = []
    generation_diagnostics = []

    def build_output() -> dict[str, Any]:
        return {
            "metadata": {
                "input": str(args.input),
                "mode": args.mode,
                "model": args.model_name,
                "offset": args.offset,
                "requested_samples": args.max_samples,
                "max_output_tokens": args.max_output_tokens,
                "records": len(records),
                "failures": len(failures),
                "generation_diagnostics": generation_diagnostics,
            },
            "records": records,
            "failures": failures,
        }

    for entity in tqdm(entities, desc=f"Gemini {args.mode}"):
        prompt = build_prompt(args.mode, entity)
        try:
            response = client.models.generate_content(
                model=args.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=args.temperature,
                    maxOutputTokens=args.max_output_tokens,
                    responseMimeType="application/json",
                ),
            )
            record = extract_json_object(response.text)
            record.setdefault("source_id", entity.get("source_id") or entity.get("id"))
            record.setdefault("code", entity.get("code", ""))
            record.setdefault("title", entity.get("title", ""))
            record.setdefault("entities", [])
            record.setdefault("relations", [])
            records.append(record)
            generation_diagnostics.append(
                {
                    "source_id": record["source_id"],
                    "code": record["code"],
                    **response_diagnostics(response, prompt),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "source_id": entity.get("source_id") or entity.get("id"),
                    "code": entity.get("code", ""),
                    "title": entity.get("title", ""),
                    "error": str(exc),
                }
            )

        if args.sleep:
            time.sleep(args.sleep)

        if args.checkpoint_every and (len(records) + len(failures)) % args.checkpoint_every == 0:
            write_json(args.output, build_output())

    write_json(args.output, build_output())
    print(f"Wrote {len(records)} records to {args.output}")
    if failures:
        print(f"Failures: {len(failures)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gemini on split ICD inputs.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=["structure", "long_text"])
    parser.add_argument("--model_name", default="gemini-2.5-flash")
    parser.add_argument("--max_samples", type=int, default=3)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_output_tokens", type=int, default=65535)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--checkpoint_every", type=int, default=1)
    parser.add_argument("--api_key")
    return parser


def main() -> None:
    run_generation(build_parser().parse_args())


if __name__ == "__main__":
    main()
