"""
Generate COT knowledge-graph extraction samples with Gemini Pro.

Output is compatible with 0413/convert_to_llamafactory.py:
[
  {
    "input": "...",
    "input_used": "...",
    "cot": "...",
    "output": {...},
    "success": true,
    "global_idx": 0,
    "input_chars": 1234,
    "original_input_chars": 1234
  }
]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm


from run_split_gemini_generation import KG_SCHEMA_PROMPT, load_json, response_diagnostics, write_json


COT_USER_TEMPLATE = """Read the following medical text and strictly follow this output format:

<think>
Write a concise reasoning trace in no more than 6 steps. Use English only. Focus on extraction decisions specific to this sample.
</think>
<output>
Write the final JSON object here.
</output>

Requirements:
0) Your response must contain exactly two top-level blocks: <think>...</think> and <output>...</output>.
1) Extract only explicitly stated information.
2) `<output>` must be strict JSON.
3) Prioritize a complete `<output>` over a longer thinking trace.
4) Use English source wording only. Do not translate terms into Chinese.
5) The relation source and target must exist in the entities.
6) Do not use markdown code fences.

Coverage requirements:
1) Prefer high coverage over short summarization.
2) Keep explicitly listed symptoms, criteria, subtypes, differential diagnoses, patient features, and risk factors whenever they appear.
3) Never infer; keep only text-supported information.

Medical text:
{text}
"""


def entity_to_medical_text(entity: dict[str, Any]) -> str:
    parts = [entity.get("title", "")]
    if entity.get("definition"):
        parts.append(entity["definition"])
    if entity.get("diagnosticCriteria"):
        parts.append(entity["diagnosticCriteria"])
    if entity.get("narrowerTerms"):
        parts.append("Narrower terms: " + json.dumps(entity["narrowerTerms"], ensure_ascii=False))
    return "\n\n".join(part for part in parts if part)


def extract_tagged_response(text: str) -> tuple[str, dict[str, Any]]:
    think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    output_match = re.search(r"<output>(.*?)</output>", text, flags=re.DOTALL | re.IGNORECASE)
    cot = think_match.group(1).strip() if think_match else ""
    output_text = output_match.group(1).strip() if output_match else text.strip()
    output_text = output_text.replace("```json", "").replace("```", "").strip()
    start = output_text.find("{")
    end = output_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in <output> block")
    return cot, json.loads(output_text[start : end + 1])


def split_response_parts(response: Any) -> tuple[str, str]:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return getattr(response, "text", "") or "", ""
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    answer_parts = []
    thought_parts = []
    for part in parts:
        text = getattr(part, "text", None)
        if not text:
            continue
        if getattr(part, "thought", False):
            thought_parts.append(text)
        else:
            answer_parts.append(text)
    answer = "".join(answer_parts).strip() or (getattr(response, "text", "") or "")
    thoughts = "\n".join(part.strip() for part in thought_parts if part.strip())
    return answer, thoughts


def run_generation(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    load_dotenv(script_dir / ".env")
    load_dotenv()

    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY. Set it in .env or pass --api_key.")

    data = load_json(args.input)
    entities = data.get("entities", [])
    selected = entities[args.offset :]
    if args.max_samples is not None:
        selected = selected[: args.max_samples]

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(args.timeout * 1000)),
    )

    samples = []
    failures = []
    diagnostics = []

    def dump_current() -> None:
        write_json(args.output, samples)
        metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
        write_json(
            metadata_path,
            {
                "input": str(args.input),
                "model": args.model_name,
                "offset": args.offset,
                "requested_samples": args.max_samples,
                "max_output_tokens": args.max_output_tokens,
                "records": len(samples),
                "failures": len(failures),
                "generation_diagnostics": diagnostics,
                "failures_detail": failures,
            },
        )

    for local_idx, entity in enumerate(tqdm(selected, desc="Gemini Pro COT")):
        global_idx = args.offset + local_idx
        model_input = entity_to_medical_text(entity)
        prompt = COT_USER_TEMPLATE.format(text=model_input)
        try:
            response = client.models.generate_content(
                model=args.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    systemInstruction=KG_SCHEMA_PROMPT,
                    temperature=args.temperature,
                    maxOutputTokens=args.max_output_tokens,
                    thinkingConfig=types.ThinkingConfig(
                        includeThoughts=args.include_thoughts,
                        thinkingBudget=args.thinking_budget,
                    ),
                ),
            )
            answer_text, thought_text = split_response_parts(response)
            cot, output = extract_tagged_response(answer_text)
            if not cot and thought_text:
                cot = thought_text
            samples.append(
                {
                    "input": model_input,
                    "input_used": model_input,
                    "cot": cot,
                    "output": output,
                    "success": True,
                    "global_idx": global_idx,
                    "source_id": entity.get("source_id") or entity.get("id"),
                    "code": entity.get("code", ""),
                    "title": entity.get("title", ""),
                    "input_chars": len(model_input),
                    "original_input_chars": len(model_input),
                    "cot_style": args.cot_style,
                    "response_had_think_tag": "<think>" in answer_text.lower(),
                    "response_had_output_tag": "<output>" in answer_text.lower(),
                    "response_had_thought_parts": bool(thought_text),
                }
            )
            diagnostics.append(
                {
                    "global_idx": global_idx,
                    "code": entity.get("code", ""),
                    **response_diagnostics(response, prompt),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "global_idx": global_idx,
                    "source_id": entity.get("source_id") or entity.get("id"),
                    "code": entity.get("code", ""),
                    "title": entity.get("title", ""),
                    "error": str(exc),
                }
            )

        if args.checkpoint_every and (len(samples) + len(failures)) % args.checkpoint_every == 0:
            dump_current()
        if args.sleep:
            time.sleep(args.sleep)

    dump_current()
    print(f"Wrote {len(samples)} COT records to {args.output}")
    if failures:
        print(f"Failures: {len(failures)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Gemini Pro COT KG samples.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model_name", default="gemini-3-pro-preview")
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_output_tokens", type=int, default=65535)
    parser.add_argument("--include_thoughts", action="store_true")
    parser.add_argument("--thinking_budget", type=int)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--checkpoint_every", type=int, default=1)
    parser.add_argument("--cot_style", choices=["template", "specific"], default="template")
    parser.add_argument("--api_key")
    return parser


def main() -> None:
    run_generation(build_parser().parse_args())


if __name__ == "__main__":
    main()
