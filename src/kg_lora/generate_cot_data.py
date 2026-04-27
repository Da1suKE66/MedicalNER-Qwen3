#!/usr/bin/env python
"""
医学知识图谱CoT数据生成器（API Key 版）
使用 google.generativeai + GEMINI_API_KEY
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv


DATA_PATH = Path(os.getenv("KG_DATA_PATH", "data/raw/mental_disorders.json"))
GEMINI_API_KEY = None

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

# Example JSON output
{
  "entities": [
    {
      "id": "D1",
      "label": "Disease",
      "name": "Major Depressive Disorder",
      "properties": {
        "DSM-5 Code": "296.2x",
        "Subtype": "Unipolar, Bipolar Depression",
        "Course Requirements": "At least 2 weeks",
        "Comorbidity Types": "Anxiety Disorders, Substance Use",
        "Prognosis Factors": "Early intervention improves prognosis"
      }
    }
  ],
  "relations": [
    {
      "source": "S1",
      "target": "D1",
      "relation_type": "Symptom–Disease",
      "relation_name": "Core Symptom Of",
      "relation": "is_core_symptom_of"
    }
  ]
}
"""

HIGH_COVERAGE_REQUIREMENTS = """Coverage requirements:
1) Prefer high coverage over short summarization.
2) Keep explicitly listed symptoms, criteria, subtypes, differential diagnoses, patient features, and risk factors whenever they appear.
3) Never infer; keep only text-supported information."""

COT_TEMPLATE_PROMPT = """<thinking>
Write a concise reasoning trace in no more than 6 steps, and keep the total length as short as possible.
</thinking>
<output>
Write the final JSON object here.
</output>

Requirements:
1) Extract only explicitly stated information.
2) `<output>` must be strict JSON.
3) Prioritize a complete `<output>` over a longer thinking trace."""

COT_SPECIFIC_PROMPT = """<thinking>
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
</thinking>
<output>
Write the final JSON object here.
</output>

Requirements:
1) Extract only explicitly stated information.
2) `<output>` must be strict JSON.
3) Prioritize a complete `<output>` over a longer thinking trace."""

def robust_json_parse(response_text: str):
    import re
    import json

    try:
        # 1. 去掉 markdown ```json ``` 包裹
        cleaned = re.sub(r"```json|```", "", response_text, flags=re.IGNORECASE)
        cleaned = re.sub(r"</?thinking>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"</?output>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", cleaned)

        # 2. 替换中文破折号
        cleaned = cleaned.replace("–", "-").replace("—", "-")

        # 3. 匹配最外层 JSON 对象
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            json_str = match.group()
            # 4. 去掉尾部多余逗号
            json_str = re.sub(r",\s*}", "}", json_str)
            json_str = re.sub(r",\s*\]", "]", json_str)
            return json.loads(json_str)
        else:
            print("⚠️ Failed to match JSON. Preview of output:")
            print(response_text[:1000])  # 打印前1000字符
            return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        print("📄 原始输出前500字符预览:")
        print(response_text[:500])
        return None
    except Exception as e:
        print(f"❌ Unexpected error in JSON parsing: {e}")
        print("📄 原始输出前500字符预览:")
        print(response_text[:500])
        return None


def extract_cot_and_json_text(response_text: str):
    text = response_text.strip()
    cot_text = ""
    json_text = text

    lower_text = text.lower()
    thinking_start_tag = "<thinking>"
    thinking_end_tag = "</thinking>"
    output_start_tag = "<output>"
    output_end_tag = "</output>"

    thinking_start = lower_text.find(thinking_start_tag)
    output_start = lower_text.find(output_start_tag)

    if thinking_start != -1:
        cot_begin = thinking_start + len(thinking_start_tag)
        thinking_end = lower_text.find(thinking_end_tag, cot_begin)
        cot_end = thinking_end
        if cot_end == -1 and output_start != -1:
            cot_end = output_start
        if cot_end == -1:
            cot_end = len(text)
        cot_text = text[cot_begin:cot_end].strip()

    if output_start != -1:
        json_begin = output_start + len(output_start_tag)
        output_end = lower_text.find(output_end_tag, json_begin)
        if output_end == -1:
            json_text = text[json_begin:].strip()
        else:
            json_text = text[json_begin:output_end].strip()
    elif thinking_start != -1:
        thinking_end = lower_text.find(thinking_end_tag, thinking_start + len(thinking_start_tag))
        if thinking_end != -1:
            json_text = text[thinking_end + len(thinking_end_tag):].strip()
        else:
            json_text = ""

    return cot_text, json_text

def classify_exception(exc: Exception) -> str:
    exc_type = type(exc).__name__
    message = str(exc)
    lowered = message.lower()

    if any(token in lowered for token in ["timed out", "timeout", "handshake", "connection", "ssl", "dns", "proxy"]):
        return f"网络错误[{exc_type}]: {message}"
    if "429" in lowered or "resource_exhausted" in lowered:
        return f"限流错误[{exc_type}]: {message}"
    if any(token in lowered for token in ["401", "403", "api key", "permission", "unauthorized", "forbidden"]):
        return f"鉴权错误[{exc_type}]: {message}"
    if "404" in lowered or "not found" in lowered:
        return f"模型或接口不存在[{exc_type}]: {message}"
    return f"API调用失败[{exc_type}]: {message}"


def init_gemini(api_key_env: str = "GEMINI_API_KEY", disable_proxy: bool = False):
    global GEMINI_API_KEY
    project_root = Path(__file__).resolve().parents[2]
    env_candidates = [
        project_root / ".env",
        Path.cwd() / ".env",
        Path.home() / ".env",
    ]
    env_path = next((p for p in env_candidates if p.exists()), None)
    if env_path is None:
        print("❌ 未找到可用的.env（已检查: 项目根目录 .env, 当前目录 .env, ~/.env）")
        sys.exit(1)

    load_dotenv(env_path)
    print(f"🔑 使用.env: {env_path}")

    if disable_proxy:
        for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
            os.environ.pop(key, None)
        print("🌐 已禁用代理环境变量（HTTP_PROXY/HTTPS_PROXY/ALL_PROXY）")

    api_key = os.getenv(api_key_env)
    if not api_key:
        print(f"❌ 未在.env中找到 {api_key_env}")
        sys.exit(1)
    GEMINI_API_KEY = api_key
    return api_key


def run_preflight_check(api_key: str, request_timeout: float, disable_proxy: bool = False):
    test_url = "https://generativelanguage.googleapis.com/v1beta/models"
    try:
        query = urlencode({"key": api_key})
        req = Request(f"{test_url}?{query}", method="GET")
        with urlopen(req, timeout=min(request_timeout, 15.0)) as resp:
            status = getattr(resp, "status", 200)
            body = resp.read(300).decode("utf-8", errors="replace")
        print(f"🧪 API预检: GET {test_url} -> HTTP {status}")
        if status >= 400:
            print(f"⚠️ 预检返回异常响应: {body.replace(chr(10), ' ')}")
    except HTTPError as e:
        preview = e.read(300).decode("utf-8", errors="replace")
        print(f"❌ API预检失败: HTTP {e.code} {e.reason}")
        print(f"   响应预览: {preview.replace(chr(10), ' ')}")
        sys.exit(1)
    except URLError as e:
        print(f"❌ API预检失败: 网络错误[URLError]: {e}")
        print("   可先尝试: 1) 加 --disable_proxy 直连  2) 检查本机代理/VPN  3) 确认 API Key 可用")
        sys.exit(1)
    except Exception as e:
        print(f"❌ API预检失败: {classify_exception(e)}")
        print("   可先尝试: 1) 加 --disable_proxy 直连  2) 检查本机代理/VPN  3) 确认 API Key 可用")
        sys.exit(1)


def extract_text_from_response(response_data: dict) -> str:
    candidates = response_data.get("candidates") or []
    if not candidates:
        return ""

    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    texts = [part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text")]
    return "\n".join(texts).strip()


def request_gemini_content(
    model_name: str,
    system_prompt: str,
    user_message: str,
    max_output_tokens: int,
    request_timeout: float,
    response_mime_type: Optional[str] = "application/json",
):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY 尚未初始化")

    endpoint_model = model_name.removeprefix("models/")
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{endpoint_model}:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_message}],
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": max_output_tokens,
        },
    }
    if response_mime_type:
        payload["generationConfig"]["responseMimeType"] = response_mime_type
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=request_timeout) as resp:
            response_data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {error_body[:500]}") from e
    except URLError as e:
        raise RuntimeError(f"URLError: {e}") from e

    if response_data.get("error"):
        raise RuntimeError(json.dumps(response_data["error"], ensure_ascii=False))

    response_text = extract_text_from_response(response_data)
    if not response_text:
        raise RuntimeError(f"响应中没有可解析文本: {json.dumps(response_data, ensure_ascii=False)[:800]}")
    return response_text


def load_data(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_records(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("entities"), list):
            return data["entities"]
        return [data]
    return [data]


def normalize_text_key(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def extract_text_from_record(record):
    if not isinstance(record, dict):
        return str(record)

    fields = [
        "title",
        "definition",
        "longDefinition",
        "diagnostic_criteria",
        "diagnosticCriteria",
        "description",
    ]
    parts = [str(record.get(k)) for k in fields if record.get(k)]
    return "\n".join(parts)


def truncate_medical_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    # `input` / `input_used` 必须与实际送给模型的文本一致，不能混入占位提示。
    # 优先在句子或段落边界截断，避免把语义单元从中间硬切开。
    window = text[:max_chars]
    boundary_candidates = [
        "\n\n",
        "\n### ",
        "\n## ",
        "\n- ",
        "\n",
        ". ",
        ".\n",
        "。",
        "；",
        "; ",
    ]

    best_cut = -1
    for marker in boundary_candidates:
        pos = window.rfind(marker)
        if pos > best_cut:
            best_cut = pos + len(marker.rstrip())

    if best_cut >= int(max_chars * 0.7):
        return window[:best_cut].rstrip()

    return window.rstrip()


def split_text_into_chunks(text: str, max_chars: int, overlap_chars: int):
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    overlap_chars = max(0, min(overlap_chars, max_chars // 3))
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(text_len, start + max_chars)
        chunk = text[start:end]

        if end < text_len:
            boundary = max(
                chunk.rfind("\n\n"),
                chunk.rfind("\n"),
                chunk.rfind(". "),
            )
            if boundary > max_chars * 0.6:
                end = start + boundary + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())
        if end >= text_len:
            break
        start = max(end - overlap_chars, start + 1)

    return [chunk for chunk in chunks if chunk]


def looks_like_truncated_json(response_text: str) -> bool:
    stripped = response_text.strip()
    return stripped.startswith("{") and stripped.count("{") > stripped.count("}")


def call_gemini_with_cot(
    medical_text: str,
    model_name: str,
    max_output_tokens: int,
    request_timeout: float,
    max_retries: int = 3,
    enable_cot: bool = False,
    cot_style: str = "specific",
):
    last_error = None
    working_text = medical_text
    for attempt in range(max_retries):
        try:
            if enable_cot:
                cot_instruction = COT_SPECIFIC_PROMPT if cot_style == "specific" else COT_TEMPLATE_PROMPT
                user_message = f"""Read the following medical text and strictly follow this output format:

{cot_instruction}
{HIGH_COVERAGE_REQUIREMENTS}

Medical text:
{working_text}"""
            else:
                user_message = f"""Perform a high-coverage extraction over the following medical text and output a structured JSON result.

Medical text:
{working_text}

Steps:
1) Mark key entities.
2) Extract entities and properties using only explicitly stated information.
3) Build only well-supported relations.
4) Verify that every relation endpoint exists in `entities`.
5) Check whether you missed explicitly listed symptoms, diagnostic criteria, differential diagnoses, subtypes, risk factors, or patient features.

{HIGH_COVERAGE_REQUIREMENTS}

Output the JSON object directly, with no markdown code blocks."""

            response_text = request_gemini_content(
                model_name=model_name,
                system_prompt=SYSTEM_PROMPT,
                user_message=user_message,
                max_output_tokens=max_output_tokens,
                request_timeout=request_timeout,
                response_mime_type=None if enable_cot else "application/json",
            )

            cot_part, json_text = extract_cot_and_json_text(response_text)
            parsed = robust_json_parse(json_text)
            if parsed is None:
                # 打印整个 response_text 供调试
                print("❌ Failed to parse JSON. Full model output preview:")
                print(response_text[:1500])  # 打印前1500字符
                if enable_cot and not json_text.strip():
                    last_error = "JSON解析失败: 模型只输出thinking，未输出JSON"
                elif looks_like_truncated_json(json_text):
                    last_error = "JSON解析失败: 模型输出被截断"
                    if attempt < max_retries - 1 and len(working_text) > 1200:
                        working_text = truncate_medical_text(working_text, max(1200, int(len(working_text) * 0.7)))
                        print(f"  ↘️ 检测到输出截断，重试时缩短输入到 {len(working_text)} 字符")
                else:
                    last_error = "JSON解析失败: 未能解析有效 JSON"
                if attempt < max_retries - 1:
                    time.sleep(2)
                continue

            return {
                "input": medical_text,
                "input_used": working_text,
                "cot": cot_part,
                "cot_style": cot_style if enable_cot else None,
                "output": parsed,
                "success": True,
            }
        except Exception as e:
            last_error = classify_exception(e)
            print(f"  ❌ {last_error[:120]}（attempt {attempt + 1}/{max_retries}）")
            if attempt < max_retries - 1:
                time.sleep(min(2 * (attempt + 1), 6))

    return {
        "input": medical_text,
        "input_used": working_text,
        "cot": None,
        "cot_style": cot_style if enable_cot else None,
        "output": None,
        "success": False,
        "error": last_error or "Max retries exceeded",
    }


def get_output_relations(output: dict):
    relations = output.get("relations")
    if isinstance(relations, list):
        return relations
    relationships = output.get("relationships")
    if isinstance(relationships, list):
        return relationships
    return []


def normalize_chunk_output(output: dict):
    entities = output.get("entities")
    if not isinstance(entities, list):
        entities = []

    prefix_counts = {}
    entities_by_key = {}
    id_to_key = {}
    normalized_entities = []

    for entity in entities:
        if not isinstance(entity, dict):
            continue

        label = str(entity.get("label", "")).strip() or "Entity"
        name = str(entity.get("name", "")).strip()
        if not name:
            continue

        entity_key = (label, normalize_text_key(name))
        existing = entities_by_key.get(entity_key)
        if existing is not None:
            existing["properties"] = merge_property_dict(existing.get("properties") or {}, entity.get("properties") or {})
            entity_id = str(entity.get("id", "")).strip()
            if entity_id:
                id_to_key[entity_id] = entity_key
            continue

        entity_id = str(entity.get("id", "")).strip()
        if not entity_id or entity_id in id_to_key:
            prefix = label_prefix(label)
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
            entity_id = f"{prefix}{prefix_counts[prefix]}"
        else:
            prefix = "".join(ch for ch in entity_id if ch.isalpha()) or label_prefix(label)
            prefix_counts[prefix] = max(prefix_counts.get(prefix, 0), 1)

        normalized = {
            "id": entity_id,
            "label": label,
            "name": name,
            "properties": merge_property_dict({}, entity.get("properties") or {}),
        }
        entities_by_key[entity_key] = normalized
        id_to_key[entity_id] = entity_key
        normalized_entities.append(normalized)

    name_to_id = {key: entity["id"] for key, entity in entities_by_key.items()}
    valid_ids = {entity["id"] for entity in normalized_entities}

    normalized_relations = []
    dropped_ghost_relations = 0
    seen_relations = set()
    for rel in get_output_relations(output):
        if not isinstance(rel, dict):
            continue

        source_raw = str(rel.get("source", "")).strip()
        target_raw = str(rel.get("target", "")).strip()
        if not source_raw or not target_raw:
            dropped_ghost_relations += 1
            continue

        source_id = source_raw if source_raw in valid_ids else None
        if not source_id:
            source_norm = normalize_text_key(source_raw)
            for (label, norm_name), candidate_id in name_to_id.items():
                if norm_name == source_norm:
                    source_id = candidate_id
                    break

        target_id = target_raw if target_raw in valid_ids else None
        if not target_id:
            target_norm = normalize_text_key(target_raw)
            for (label, norm_name), candidate_id in name_to_id.items():
                if norm_name == target_norm:
                    target_id = candidate_id
                    break

        if not source_id or not target_id:
            dropped_ghost_relations += 1
            continue

        relation_record = {
            "source": source_id,
            "target": target_id,
            "relation_type": str(rel.get("relation_type", "")).strip(),
            "relation_name": str(rel.get("relation_name", "")).strip(),
            "relation": str(rel.get("relation", "")).strip(),
        }
        relation_key = (
            relation_record["source"],
            relation_record["target"],
            relation_record["relation_type"],
            relation_record["relation_name"],
            relation_record["relation"],
        )
        if relation_key in seen_relations:
            continue
        seen_relations.add(relation_key)
        normalized_relations.append(relation_record)

    return {
        "entities": normalized_entities,
        "relations": normalized_relations,
    }, {
        "entity_count": len(normalized_entities),
        "relation_count": len(normalized_relations),
        "dropped_ghost_relations": dropped_ghost_relations,
    }


def merge_property_dict(base: dict, incoming: dict):
    merged = dict(base or {})
    for key, value in (incoming or {}).items():
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue

        existing = str(merged.get(key, "")).strip()
        if not existing:
            merged[key] = text
        elif text != existing and text not in existing.split(" | "):
            merged[key] = f"{existing} | {text}"
    return merged


def label_prefix(label: str) -> str:
    mapping = {
        "Disease": "D",
        "Symptom": "S",
        "Diagnostic Criteria": "DC",
        "Interview Tool": "IT",
        "Patient Information": "PI",
        "Medication": "M",
        "Communication Method": "CM",
        "Risk Information": "R",
    }
    return mapping.get(label, "E")


def merge_chunk_outputs(chunk_outputs):
    entities_by_key = {}
    local_to_global = {}
    global_name_to_id = {}
    prefix_counts = {}
    merged_relations = []
    seen_relations = set()

    for chunk_idx, output in enumerate(chunk_outputs):
        chunk_entities = output.get("entities") or []
        chunk_entity_lookup = {}

        for entity in chunk_entities:
            if not isinstance(entity, dict):
                continue

            label = str(entity.get("label", "")).strip() or "Entity"
            name = str(entity.get("name", "")).strip()
            if not name:
                continue

            entity_key = (label, normalize_text_key(name))
            existing = entities_by_key.get(entity_key)
            if existing is None:
                prefix = label_prefix(label)
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
                global_id = f"{prefix}{prefix_counts[prefix]}"
                existing = {
                    "id": global_id,
                    "label": label,
                    "name": name,
                    "properties": merge_property_dict({}, entity.get("properties") or {}),
                }
                entities_by_key[entity_key] = existing
                global_name_to_id[(label, normalize_text_key(name))] = global_id
            else:
                existing["properties"] = merge_property_dict(existing.get("properties") or {}, entity.get("properties") or {})

            local_id = str(entity.get("id", "")).strip()
            if local_id:
                local_to_global[(chunk_idx, local_id)] = existing["id"]
            chunk_entity_lookup[normalize_text_key(name)] = existing["id"]

        for rel in get_output_relations(output):
            if not isinstance(rel, dict):
                continue

            rel_source = str(rel.get("source", "")).strip()
            rel_target = str(rel.get("target", "")).strip()
            source_id = local_to_global.get((chunk_idx, rel_source))
            target_id = local_to_global.get((chunk_idx, rel_target))

            if not source_id and rel_source:
                source_norm = normalize_text_key(rel_source)
                source_id = chunk_entity_lookup.get(source_norm)
                if not source_id:
                    for (label, norm_name), global_id in global_name_to_id.items():
                        if norm_name == source_norm:
                            source_id = global_id
                            break

            if not target_id and rel_target:
                target_norm = normalize_text_key(rel_target)
                target_id = chunk_entity_lookup.get(target_norm)
                if not target_id:
                    for (label, norm_name), global_id in global_name_to_id.items():
                        if norm_name == target_norm:
                            target_id = global_id
                            break

            if not source_id or not target_id:
                continue

            relation_record = {
                "source": source_id,
                "target": target_id,
                "relation_type": rel.get("relation_type", ""),
                "relation_name": rel.get("relation_name", ""),
                "relation": rel.get("relation", ""),
            }
            relation_key = (
                relation_record["source"],
                relation_record["target"],
                relation_record["relation_type"],
                relation_record["relation_name"],
                relation_record["relation"],
            )
            if relation_key in seen_relations:
                continue
            seen_relations.add(relation_key)
            merged_relations.append(relation_record)

    merged_entities = sorted(entities_by_key.values(), key=lambda x: x["id"])
    merged_relations.sort(key=lambda x: (x["source"], x["target"], x["relation"], x["relation_name"]))
    return {
        "entities": merged_entities,
        "relations": merged_relations,
    }


def repair_chunk_merged_output(text: str, merged_output: dict, chunk_traces, args):
    sanitized_output, pre_stats = normalize_chunk_output(merged_output)
    failed_chunks = [
        trace.get("chunk_index")
        for trace in chunk_traces
        if isinstance(trace, dict) and trace.get("success") is False
    ]

    repair_prompt = f"""You are repairing a chunk-merged medical knowledge graph built from the full source text.

Primary objective:
- Perform the MINIMUM necessary repair to make the merged graph more complete and structurally valid.

Repair policy:
1) Preserve existing valid entities and valid relations whenever possible.
2) Do NOT rewrite the graph from scratch.
3) Do NOT rename existing entities unless the name is clearly malformed or duplicated with the same meaning.
4) Do NOT delete valid entities just to simplify the graph.
5) Only add entities or relations when they are explicitly supported by the full source text and were likely missed across chunks.
6) If a relation endpoint does not resolve, first try to map it to an existing entity with the same meaning; only if impossible, drop that relation.
7) If a chunk failed, use the full text to recover only the clearly explicit missing information.
8) Favor minimal edits over aggressive re-organization.

Output requirements:
- Output one consolidated graph for the FULL text, not per chunk.
- Every entity must have: id, label, name, properties
- Every relation must have: source, target, relation_type, relation_name, relation
- Every relation source/target must reference an entity id that exists in entities
- Use only information explicitly stated in the source text
- Output strict JSON only

What to avoid:
- Do not replace most ids with a totally new numbering scheme unless required for consistency.
- Do not drop large parts of the current graph if they are already valid.
- Do not infer hidden medical knowledge.
- Do not compress multiple distinct explicit entities into one unless they are obvious duplicates.

Chunk merge diagnostics:
- successful_chunks: {sum(1 for trace in chunk_traces if isinstance(trace, dict) and trace.get("success"))}
- failed_chunks: {len(failed_chunks)}
- failed_chunk_indexes: {failed_chunks}
- current_entity_count: {pre_stats['entity_count']}
- current_relation_count: {pre_stats['relation_count']}
- dropped_ghost_relations_before_repair: {pre_stats['dropped_ghost_relations']}

Source text:
{text}

Current merged graph draft:
{json.dumps(sanitized_output, ensure_ascii=False, indent=2)}
"""

    last_error = None
    for attempt in range(args.max_retries):
        try:
            response_text = request_gemini_content(
                model_name=args.model_name,
                system_prompt=SYSTEM_PROMPT,
                user_message=repair_prompt,
                max_output_tokens=args.max_output_tokens,
                request_timeout=args.request_timeout,
                response_mime_type="application/json",
            )
            parsed = robust_json_parse(response_text)
            if parsed is None:
                last_error = "repair pass JSON解析失败"
                if attempt < args.max_retries - 1:
                    time.sleep(min(2 * (attempt + 1), 6))
                continue

            repaired_output, repair_stats = normalize_chunk_output(parsed)
            return repaired_output, {
                "attempted": True,
                "success": True,
                "attempts": attempt + 1,
                "failed_chunk_indexes": failed_chunks,
                "pre_repair_stats": pre_stats,
                "post_repair_stats": repair_stats,
            }
        except Exception as e:
            last_error = classify_exception(e)
            print(f"  ❌ repair pass失败: {last_error[:120]}（attempt {attempt + 1}/{args.max_retries}）")
            if attempt < args.max_retries - 1:
                time.sleep(min(2 * (attempt + 1), 6))

    return sanitized_output, {
        "attempted": True,
        "success": False,
        "attempts": args.max_retries,
        "failed_chunk_indexes": failed_chunks,
        "pre_repair_stats": pre_stats,
        "post_repair_stats": pre_stats,
        "error": last_error or "repair pass failed",
    }


def process_record_chunked(record, global_idx, args):
    text = extract_text_from_record(record)
    if not text or not text.strip():
        return {"success": False, "skip": True, "error": "文本过短", "global_idx": global_idx}

    chunks = split_text_into_chunks(text, args.max_input_chars, args.chunk_overlap_chars)
    chunk_outputs = []
    chunk_traces = []
    chunk_errors = []

    print(f"  🧩 record[{global_idx}] 分段抽取: {len(chunks)} chunks")
    for chunk_idx, chunk_text in enumerate(chunks, 1):
        chunk_result = call_gemini_with_cot(
            chunk_text,
            model_name=args.model_name,
            max_output_tokens=args.max_output_tokens,
            request_timeout=args.request_timeout,
            max_retries=args.max_retries,
            enable_cot=args.enable_cot,
            cot_style=args.cot_style,
        )
        if chunk_result.get("success"):
            chunk_outputs.append(chunk_result["output"])
            chunk_traces.append(
                {
                    "chunk_index": chunk_idx,
                    "success": True,
                    "input_used": chunk_result.get("input_used", chunk_text),
                    "cot": chunk_result.get("cot"),
                    "cot_style": chunk_result.get("cot_style"),
                    "output": chunk_result.get("output"),
                }
            )
            print(f"    - chunk {chunk_idx}/{len(chunks)} ✅")
        else:
            error_payload = {
                "chunk_index": chunk_idx,
                "success": False,
                "input_used": chunk_result.get("input_used", chunk_text),
                "cot": chunk_result.get("cot"),
                "cot_style": chunk_result.get("cot_style"),
                "output": chunk_result.get("output"),
                "error": chunk_result.get("error", "Unknown"),
            }
            chunk_errors.append(error_payload)
            chunk_traces.append(error_payload)
            print(f"    - chunk {chunk_idx}/{len(chunks)} ❌ {chunk_result.get('error', 'Unknown')}")

    if not chunk_outputs:
        return {
            "success": False,
            "global_idx": global_idx,
            "error": "所有chunks都失败",
            "chunk_errors": chunk_errors,
            "chunk_traces": chunk_traces,
            "input_chars": min(len(text), args.max_input_chars),
            "original_input_chars": len(text),
        }

    merged_output = merge_chunk_outputs(chunk_outputs)
    repaired_output, repair_meta = repair_chunk_merged_output(text, merged_output, chunk_traces, args)
    return {
        "input": text,
        "input_used": chunks,
        "cot": None,
        "cot_style": args.cot_style if args.enable_cot else None,
        "output": repaired_output,
        "success": True,
        "extraction_mode": "chunk_merge",
        "global_idx": global_idx,
        "input_chars": sum(len(chunk) for chunk in chunks),
        "original_input_chars": len(text),
        "chunk_count": len(chunks),
        "successful_chunks": len(chunk_outputs),
        "failed_chunks": len(chunk_errors),
        "chunk_errors": chunk_errors,
        "chunk_traces": chunk_traces,
        "repair_pass": repair_meta,
    }


def process_record(record, global_idx, args):
    if args.extraction_mode == "chunk_merge":
        return process_record_chunked(record, global_idx, args)

    text = extract_text_from_record(record)
    if not text or len(text) < args.min_text_length:
        return {"success": False, "skip": True, "error": "文本过短", "global_idx": global_idx}

    original_length = len(text)
    text = truncate_medical_text(text, args.max_input_chars)

    result = call_gemini_with_cot(
        text,
        model_name=args.model_name,
        max_output_tokens=args.max_output_tokens,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        enable_cot=args.enable_cot,
        cot_style=args.cot_style,
    )
    result["global_idx"] = global_idx
    result["input_chars"] = len(text)
    result["original_input_chars"] = original_length
    return result


def generate_cot_data(records, args, start_idx=0, max_samples=None, shard_name=None):
    total = min(max_samples, len(records) - start_idx) if max_samples else len(records) - start_idx
    prefix = f"[{shard_name}] " if shard_name else ""
    print(f"\n{prefix}🚀 开始生成（共{total}条，从{start_idx}开始）")

    target = records[start_idx:start_idx + total]
    generated = []
    done = 0
    start_time = time.time()
    success = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=1) as ex:
        futures = {
            ex.submit(process_record, rec, start_idx + i, args): i + 1
            for i, rec in enumerate(target)
        }
        for future in as_completed(futures):
            idx = futures[future]
            done += 1
            try:
                result = future.result()
            except Exception as e:
                print(f"  {prefix}❌ [{idx}/{total}] 线程异常: {e}")
                continue

            if result.get("skip"):
                skipped += 1
                print(f"  {prefix}⊘ [{idx}/{total}] 文本过短")
            elif result.get("success"):
                success += 1
                generated.append(result)
                print(f"  {prefix}✅ [{idx}/{total}] 成功")
            else:
                generated.append(result)
                print(f"  {prefix}❌ [{idx}/{total}] 失败: {result.get('error', 'Unknown')}")

            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            remain = total - done
            eta = int(remain / rate) if rate > 0 else -1
            eta_text = f"{eta//60}m{eta%60}s" if eta >= 0 else "N/A"
            print(f"  {prefix}📈 进度: {done}/{total} ({done * 100 / total:.1f}%), ETA≈{eta_text}")

    generated.sort(key=lambda x: x.get("global_idx", 0))
    print(f"\n{prefix}✅ 完成：成功 {success}/{total}，跳过 {skipped}")
    return generated


def filter_quality_data(generated_data):
    print(f"\n🔍 质量筛查（共{len(generated_data)}条）")
    if not generated_data:
        print("✅ 保留 0/0 (0%)")
        return []

    filtered = []
    reject_reasons = {}
    for sample in generated_data:
        if not sample.get("success"):
            reject_reasons["API调用失败"] = reject_reasons.get("API调用失败", 0) + 1
            continue

        output = sample.get("output") or {}
        if sample.get("extraction_mode") == "chunk_merge":
            normalized_output, stats = normalize_chunk_output(output)
            sample["output"] = normalized_output
            quality_flags = []
            repair_pass = sample.get("repair_pass") or {}
            if stats["entity_count"] < 3:
                quality_flags.append("实体数<3")
            if stats["dropped_ghost_relations"] > 0:
                quality_flags.append(f"已清理幽灵关系:{stats['dropped_ghost_relations']}")
            if repair_pass.get("attempted") and not repair_pass.get("success"):
                quality_flags.append("repair_pass_failed")
            if quality_flags:
                sample["quality_flags"] = quality_flags
            filtered.append(sample)
            continue

        entities = output.get("entities", [])
        relations = get_output_relations(output)
        names = {e.get("name") for e in entities if isinstance(e, dict)}
        ids = {e.get("id") for e in entities if isinstance(e, dict)}

        passed = True
        reason = None
        if len(entities) < 3:
            passed = False
            reason = "实体数<3"

        if passed:
            for rel in relations:
                source = rel.get("source")
                target = rel.get("target")
                source_ok = source in names or source in ids
                target_ok = target in names or target in ids
                if not source_ok or not target_ok:
                    passed = False
                    reason = "幽灵关系"
                    break

        if passed:
            filtered.append(sample)
        else:
            reject_reasons[reason or "其他"] = reject_reasons.get(reason or "其他", 0) + 1

    pct = 100 * len(filtered) // len(generated_data)
    print(f"✅ 保留 {len(filtered)}/{len(generated_data)} ({pct}%)")
    return filtered


def save_results(generated_data, filtered_data, output_dir=None, file_suffix=None):
    output_dir = Path(output_dir) if output_dir else Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{file_suffix}" if file_suffix else ""

    raw_file = output_dir / f"cot_raw_{ts}{suffix}.json"
    filtered_file = output_dir / f"cot_filtered_{ts}{suffix}.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=2)
    with open(filtered_file, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    print(f"💾 raw: {raw_file}")
    print(f"💾 filtered: {filtered_file}")
    return raw_file, filtered_file


def split_ranges(total, parts):
    parts = max(1, min(parts, total))
    base, rem = divmod(total, parts)
    ranges = []
    s = 0
    for i in range(parts):
        n = base + (1 if i < rem else 0)
        e = s + n
        ranges.append((s, e))
        s = e
    return ranges


def run_parallel_shards(records, args):
    total = min(args.max_samples, len(records) - args.start_idx) if args.max_samples else len(records) - args.start_idx
    if total <= 0:
        print("❌ 没有可处理数据")
        return [], []

    ranges = split_ranges(total, args.workers)
    print(f"\n🧩 分为 {len(ranges)} 片并行")
    all_generated = []
    shard_files = []

    with ThreadPoolExecutor(max_workers=len(ranges)) as ex:
        future_map = {}
        for part_idx, (ls, le) in enumerate(ranges, 1):
            st = args.start_idx + ls
            size = le - ls
            print(f"  - part{part_idx}: 全局[{st}, {st + size}) 共{size}条")
            fut = ex.submit(generate_cot_data, records, args, st, size, f"part{part_idx}")
            future_map[fut] = part_idx

        for fut in as_completed(future_map):
            part_idx = future_map[fut]
            part_generated = fut.result()
            part_filtered = filter_quality_data(part_generated)
            raw_file, filtered_file = save_results(
                part_generated,
                part_filtered,
                output_dir=args.output_dir,
                file_suffix=f"part{part_idx}",
            )
            shard_files.append((part_idx, raw_file, filtered_file))
            all_generated.extend(part_generated)

    all_generated.sort(key=lambda x: x.get("global_idx", 0))
    all_filtered = filter_quality_data(all_generated)
    merged_raw, merged_filtered = save_results(
        all_generated, all_filtered, output_dir=args.output_dir, file_suffix="merged"
    )
    print("\n📦 分片文件：")
    for part_idx, raw_file, filtered_file in sorted(shard_files, key=lambda x: x[0]):
        print(f"  - part{part_idx} raw: {raw_file}")
        print(f"  - part{part_idx} filtered: {filtered_file}")
    print(f"  - merged raw: {merged_raw}")
    print(f"  - merged filtered: {merged_filtered}")
    return all_generated, all_filtered


def parse_args():
    p = argparse.ArgumentParser(description="医学知识图谱CoT数据生成器（API Key 版）")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--extraction_mode", choices=["single_pass", "chunk_merge"], default="single_pass")
    p.add_argument("--model_name", type=str, default="gemini-3-flash-preview")
    p.add_argument("--max_output_tokens", type=int, default=4000)
    p.add_argument("--max_input_chars", type=int, default=15000)
    p.add_argument("--chunk_overlap_chars", type=int, default=200)
    p.add_argument("--request_timeout", type=float, default=180.0)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--min_text_length", type=int, default=50)
    p.add_argument("--api_key_env", type=str, default="GEMINI_API_KEY")
    p.add_argument("--enable_cot", action="store_true", help="要求模型输出 <thinking> 和 <output>，并保存 cot 字段")
    p.add_argument("--cot_style", choices=["template", "specific"], default="specific", help="显式 CoT 的风格：固定模板或样本特异性")
    p.add_argument("--disable_proxy", action="store_true", help="禁用当前 shell 中的代理环境变量后再请求 API")
    p.add_argument("--skip_preflight", action="store_true", help="跳过调用前的网络预检")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("医学知识图谱CoT数据生成器 (API Key)")
    print("=" * 60)

    api_key = init_gemini(args.api_key_env, disable_proxy=args.disable_proxy)
    if not args.skip_preflight:
        run_preflight_check(api_key, args.request_timeout, disable_proxy=args.disable_proxy)

    if not DATA_PATH.exists():
        print(f"❌ 数据文件不存在: {DATA_PATH}")
        sys.exit(1)

    print(f"\n📂 加载数据: {DATA_PATH}")
    data = load_data(DATA_PATH)
    records = normalize_records(data)
    print(f"✅ 加载成功，共{len(records)}条记录")

    if args.workers <= 1:
        generated = generate_cot_data(records, args, args.start_idx, args.max_samples)
        filtered = filter_quality_data(generated)
        save_results(generated, filtered, output_dir=args.output_dir)
    else:
        run_parallel_shards(records, args)

    print("\n" + "=" * 60)
    print("✅ 任务完成！")
    print("=" * 60)
