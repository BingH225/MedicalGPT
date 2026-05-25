#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a closed-loop medical dataset bundle for MedicalGPT.

The pipeline turns public or local medical QA data into:
1. canonical intermediate cases
2. ShareGPT SFT data
3. chosen/rejected preference data for RM + DPO
4. a fixed structured-case evaluation set

The script is intentionally conservative:
- local JSONL smoke mode works with only the Python standard library
- Hugging Face dataset loading is optional and only enabled when requested
- generated data is deterministic for a given seed
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import textwrap
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROFILE_DEFAULTS = {
    "smoke": {"sft": 1000, "preference": 500, "eval": 100},
    "formal": {"sft": 6000, "preference": 3000, "eval": 400},
}

DEFAULT_LOCAL_SFT_SOURCES = [
    "data/sft/medical_sft_1K_format.jsonl",
]

DEFAULT_LOCAL_PREFERENCE_SOURCES = [
    "data/reward/dpo_zh_500.jsonl",
]

DEFAULT_HF_REPO_SFT_SOURCES = [
    {
        "repo_id": "shibing624/medical",
        "path": "finetune/train_zh_0.json",
        "source_name": "shibing624_medical_train_zh_0",
    },
    {
        "repo_id": "FreedomIntelligence/HuatuoGPT-sft-data-v1",
        "path": "HuatuoGPT_sft_data_v1.jsonl",
        "source_name": "huatuogpt_sft_data_v1",
    },
]

MEDICAL_STOPWORDS = {
    "怎么",
    "怎么办",
    "一下",
    "一个",
    "一些",
    "这种",
    "这个",
    "那个",
    "患者",
    "病人",
    "需要",
    "建议",
    "说明",
    "考虑",
    "如果",
    "以及",
    "可以",
    "目前",
    "出现",
    "可能",
    "相关",
}

ADVERTISEMENT_KEYWORDS = (
    "加微信",
    "公众号",
    "扫码",
    "私信",
    "VX",
    "v信",
    "点击链接",
    "优惠",
)

MEDICAL_KEYWORDS = (
    "发烧",
    "咳嗽",
    "疼",
    "痛",
    "腹泻",
    "呕吐",
    "头晕",
    "胸闷",
    "呼吸",
    "血压",
    "心率",
    "月经",
    "孕",
    "药",
    "检查",
    "化验",
    "症状",
    "就医",
    "诊断",
)

DEPARTMENT_RULES = [
    ("儿科", ("儿童", "小孩", "婴儿", "宝宝", "幼儿", "儿科")),
    ("妇科", ("月经", "经期", "孕", "妊娠", "妇科", "白带")),
    ("呼吸科", ("咳嗽", "咽痛", "呼吸", "鼻塞", "肺", "感冒")),
    ("消化科", ("腹痛", "腹泻", "恶心", "呕吐", "胃", "肠")),
    ("心内科", ("胸痛", "胸闷", "心慌", "心悸", "血压", "心率")),
    ("神经内科", ("头痛", "头晕", "抽搐", "麻木", "意识")),
    ("皮肤科", ("皮疹", "瘙痒", "红斑", "过敏", "皮肤")),
    ("骨科", ("腰痛", "关节", "扭伤", "骨折", "肿胀")),
    ("泌尿科", ("尿频", "尿急", "尿痛", "肾", "前列腺")),
    ("内分泌科", ("糖尿病", "血糖", "甲状腺", "肥胖")),
]

HIGH_RISK_FLAGS = {
    "胸痛": "胸痛/心血管风险",
    "呼吸困难": "呼吸困难",
    "昏迷": "意识障碍",
    "抽搐": "抽搐",
    "便血": "消化道出血",
    "黑便": "消化道出血",
    "呕血": "消化道出血",
    "高热": "持续高热",
    "39": "高热或体温偏高",
    "40": "极高热",
    "孕": "孕期特殊人群",
    "婴儿": "婴幼儿特殊人群",
    "新生儿": "新生儿特殊人群",
    "呼吸急促": "呼吸困难",
    "意识不清": "意识障碍",
    "剧烈腹痛": "急腹症风险",
    "一侧肢体无力": "脑卒中风险",
}

MODERATE_RISK_KEYWORDS = (
    "持续",
    "反复",
    "加重",
    "高血压",
    "糖尿病",
    "老人",
    "老年",
)

SAFE_QUALIFIERS = (
    "建议就医",
    "建议线下",
    "必要时急诊",
    "尽快就诊",
    "仅供参考",
    "不能替代医生面诊",
)

UNSAFE_PATTERNS = (
    "无需就医",
    "一定是",
    "可以自行长期服用",
    "不用检查",
    "百分百",
    "马上服用抗生素",
    "无需处理",
)

SECTION_SPECS = [
    ("initial_judgement", "初步判断"),
    ("key_evidence", "关键依据"),
    ("risk_signals", "风险信号"),
    ("suggested_action", "建议动作"),
    ("medication_caution", "用药/就医注意事项"),
]


@dataclass
class CanonicalCase:
    case_id: str
    source: str
    source_type: str
    department: str
    difficulty: str
    patient_profile: Dict[str, str]
    chief_complaint: str
    history_present_illness: str
    past_history: str
    question: str
    reference_answer: str
    safety_flags: List[str] = field(default_factory=list)
    coverage_tags: List[str] = field(default_factory=list)
    track: str = "structured_case"


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def stable_hash(*parts: str, length: int = 12) -> str:
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
    return digest[:length]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def is_medical_text(text: str) -> bool:
    return any(keyword in text for keyword in MEDICAL_KEYWORDS)


def looks_like_advertisement(text: str) -> bool:
    return any(keyword.lower() in text.lower() for keyword in ADVERTISEMENT_KEYWORDS)


def split_sentences(text: str, limit: int = 3) -> List[str]:
    parts = re.split(r"[。！？!?；;\n]+", normalize_text(text))
    return [part.strip() for part in parts if part.strip()][:limit]


def summarize_for_judgement(answer: str, department: str) -> str:
    sentences = split_sentences(answer, limit=2)
    if sentences:
        summary = "；".join(sentences)
    else:
        summary = f"结合当前描述，优先考虑与{department}相关的问题，但仍需结合线下检查进一步确认。"
    if "建议" not in summary and "考虑" not in summary:
        summary = "结合当前描述，" + summary
    return summary


def extract_age(question: str) -> Tuple[str, str]:
    patterns = [
        (r"(\d{1,2})岁", "岁"),
        (r"(\d{1,2})个月", "个月"),
        (r"(\d{1,2})月龄", "月龄"),
    ]
    for pattern, suffix in patterns:
        match = re.search(pattern, question)
        if match:
            raw = match.group(0)
            value = int(match.group(1))
            if suffix in ("个月", "月龄") or value <= 12:
                return raw, "儿童"
            if value >= 60:
                return raw, "老年"
            return raw, "成人"
    if any(keyword in question for keyword in ("婴儿", "宝宝", "小孩", "儿童", "幼儿")):
        return "未明确", "儿童"
    if any(keyword in question for keyword in ("老人", "老年")):
        return "未明确", "老年"
    return "未说明", "未知"


def extract_sex(question: str) -> str:
    if any(keyword in question for keyword in ("男", "男性", "先生")):
        return "男"
    if any(keyword in question for keyword in ("女", "女性", "女士")):
        return "女"
    return "未说明"


def extract_special_population(question: str) -> str:
    if any(keyword in question for keyword in ("孕妇", "怀孕", "妊娠")):
        return "孕产期"
    if any(keyword in question for keyword in ("婴儿", "宝宝", "儿童", "小孩", "幼儿")):
        return "儿科"
    return "无"


def infer_department(text: str) -> str:
    for department, keywords in DEPARTMENT_RULES:
        if any(keyword in text for keyword in keywords):
            return department
    return "全科"


def infer_safety_flags(text: str) -> List[str]:
    flags: List[str] = []
    for keyword, label in HIGH_RISK_FLAGS.items():
        if keyword in text and label not in flags:
            flags.append(label)
    return flags


def infer_difficulty(question: str, answer: str, safety_flags: Sequence[str]) -> str:
    joined = question + "\n" + answer
    if safety_flags:
        return "high"
    if any(keyword in joined for keyword in MODERATE_RISK_KEYWORDS):
        return "medium"
    return "low"


def infer_chief_complaint(question: str) -> str:
    head = split_sentences(question, limit=1)
    if head:
        return head[0]
    return question[:80]


def infer_past_history(question: str) -> str:
    keywords = [keyword for keyword in ("高血压", "糖尿病", "冠心病", "哮喘", "既往", "过敏", "手术") if keyword in question]
    if not keywords:
        return "未提供明确既往史。"
    return "提及既往相关信息：" + "、".join(keywords) + "。"


def build_patient_profile(question: str) -> Dict[str, str]:
    age_raw, age_group = extract_age(question)
    return {
        "sex": extract_sex(question),
        "age_raw": age_raw,
        "age_group": age_group,
        "special_population": extract_special_population(question),
    }


def quality_filter(question: str, answer: str) -> bool:
    question = normalize_text(question)
    answer = normalize_text(answer)
    if not question or not answer:
        return False
    if len(question) < 6 or len(answer) < 12:
        return False
    if len(question) > 1200 or len(answer) > 4000:
        return False
    if looks_like_advertisement(question + "\n" + answer):
        return False
    if not is_medical_text(question + "\n" + answer):
        return False
    return True


def canonical_case_from_qa(question: str, answer: str, source: str, source_id: str, source_type: str) -> CanonicalCase:
    question = normalize_text(question)
    answer = normalize_text(answer)
    combined = question + "\n" + answer
    department = infer_department(combined)
    safety_flags = infer_safety_flags(combined)
    difficulty = infer_difficulty(question, answer, safety_flags)
    patient_profile = build_patient_profile(question)
    coverage_tags = [department, difficulty]
    coverage_tags.extend(flag for flag in safety_flags if flag not in coverage_tags)
    return CanonicalCase(
        case_id=f"{source_id}-{stable_hash(question, answer)}",
        source=source,
        source_type=source_type,
        department=department,
        difficulty=difficulty,
        patient_profile=patient_profile,
        chief_complaint=infer_chief_complaint(question),
        history_present_illness=question,
        past_history=infer_past_history(question),
        question=question,
        reference_answer=answer,
        safety_flags=safety_flags,
        coverage_tags=coverage_tags,
    )


def extract_turn_text(turn: Dict[str, Any]) -> str:
    return normalize_text(turn.get("value") or turn.get("content") or turn.get("text"))


def extract_last_dialog_pair(messages: Sequence[Dict[str, Any]]) -> Tuple[str, str]:
    last_human = ""
    last_gpt = ""
    for turn in messages:
        role = (turn.get("from") or turn.get("role") or "").lower()
        value = extract_turn_text(turn)
        if not value:
            continue
        if role in ("human", "user"):
            last_human = value
        elif role in ("gpt", "assistant"):
            last_gpt = value
    return last_human, last_gpt


def select_text(record: Dict[str, Any], candidates: Sequence[str]) -> str:
    for name in candidates:
        value = record.get(name)
        if isinstance(value, str) and normalize_text(value):
            return normalize_text(value)
    return ""


def text_from_nested_record(record: Dict[str, Any]) -> Tuple[str, str]:
    if isinstance(record.get("conversations"), list):
        return extract_last_dialog_pair(record["conversations"])
    if isinstance(record.get("messages"), list):
        return extract_last_dialog_pair(record["messages"])
    if isinstance(record.get("data"), list):
        parts = [normalize_text(item) for item in record["data"] if normalize_text(item)]
        qa_pairs: List[Tuple[str, str]] = []
        for index in range(0, len(parts) - 1, 2):
            question = re.sub(r"^(问|Q)[：:]\s*", "", parts[index]).strip()
            answer = re.sub(r"^(答|A)[：:]\s*", "", parts[index + 1]).strip()
            if question and answer:
                qa_pairs.append((question, answer))
        if qa_pairs:
            return qa_pairs[-1]
    question = select_text(record, ("question", "query", "prompt", "input"))
    answer = select_text(record, ("answer", "response", "output", "target"))
    instruction = select_text(record, ("instruction",))
    if instruction and question and instruction != question:
        question = instruction + "\n\n" + question
    elif instruction and not question:
        question = instruction
    return question, answer


def build_cases_from_local_sft_file(path: Path, source_name: str) -> List[CanonicalCase]:
    rows = load_jsonl(path)
    cases: List[CanonicalCase] = []
    for index, row in enumerate(rows):
        question, answer = text_from_nested_record(row)
        if not quality_filter(question, answer):
            continue
        cases.append(
            canonical_case_from_qa(
                question=question,
                answer=answer,
                source=source_name,
                source_id=f"{source_name}-{index}",
                source_type="sft_source",
            )
        )
    return cases


def load_hf_records(dataset_name: str, split: str, max_records: int = -1) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Loading Hugging Face datasets requires `datasets`. "
            "Install it with `pip install datasets>=2.14.6`."
        ) from exc

    dataset = load_dataset(dataset_name, split=split)
    if max_records and max_records > 0:
        dataset = dataset.select(range(min(max_records, len(dataset))))
    return [dataset[i] for i in range(len(dataset))]


def download_hf_dataset_file(repo_id: str, path_in_repo: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Downloading Hugging Face dataset files requires `huggingface_hub`. "
            "Install it with `pip install huggingface_hub`."
        ) from exc

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=path_in_repo,
        repo_type="dataset",
    )
    return Path(downloaded)


def build_cases_from_hf_repo_file(repo_id: str, path_in_repo: str, source_name: str, max_records: int = -1) -> List[CanonicalCase]:
    local_file = download_hf_dataset_file(repo_id, path_in_repo)
    cases: List[CanonicalCase] = []
    for index, row in enumerate(iter_jsonl(local_file)):
        question, answer = text_from_nested_record(row)
        if not quality_filter(question, answer):
            continue
        cases.append(
            canonical_case_from_qa(
                question=question,
                answer=answer,
                source=source_name,
                source_id=f"{source_name}-{index}",
                source_type="hf_repo_file",
            )
        )
        if max_records > 0 and len(cases) >= max_records:
            break
    return cases


def build_cases_from_hf_dataset(dataset_name: str, split: str, max_records: int = -1) -> List[CanonicalCase]:
    rows = load_hf_records(dataset_name, split, max_records=max_records)
    cases: List[CanonicalCase] = []
    for index, row in enumerate(rows):
        question, answer = text_from_nested_record(row)
        if not quality_filter(question, answer):
            continue
        cases.append(
            canonical_case_from_qa(
                question=question,
                answer=answer,
                source=dataset_name,
                source_id=f"{dataset_name}-{index}",
                source_type="hf_source",
            )
        )
    return cases


def deduplicate_cases(cases: Sequence[CanonicalCase]) -> List[CanonicalCase]:
    seen: Dict[str, CanonicalCase] = {}
    for case in cases:
        key = stable_hash(case.question)
        current = seen.get(key)
        if current is None or len(case.reference_answer) > len(current.reference_answer):
            seen[key] = case
    return list(seen.values())


def round_robin_sample(cases: Sequence[CanonicalCase], limit: int, seed: int) -> List[CanonicalCase]:
    if limit <= 0 or not cases:
        return []
    buckets: Dict[Tuple[str, str], List[CanonicalCase]] = defaultdict(list)
    for case in cases:
        buckets[(case.department, case.difficulty)].append(case)
    rng = random.Random(seed)
    for bucket_cases in buckets.values():
        rng.shuffle(bucket_cases)
    ordered_keys = sorted(buckets.keys(), key=lambda item: f"{item[0]}::{item[1]}")
    selected: List[CanonicalCase] = []
    while ordered_keys and len(selected) < limit:
        next_keys: List[Tuple[str, str]] = []
        for key in ordered_keys:
            bucket_cases = buckets[key]
            if not bucket_cases:
                continue
            selected.append(bucket_cases.pop())
            if len(selected) >= limit:
                break
            if bucket_cases:
                next_keys.append(key)
        ordered_keys = next_keys
    return selected


def remove_cases(source: Sequence[CanonicalCase], to_remove: Sequence[CanonicalCase]) -> List[CanonicalCase]:
    remove_ids = {case.case_id for case in to_remove}
    return [case for case in source if case.case_id not in remove_ids]


def render_patient_profile(profile: Dict[str, str]) -> str:
    lines = [
        f"- 性别：{profile.get('sex', '未说明')}",
        f"- 年龄信息：{profile.get('age_raw', '未说明')}（{profile.get('age_group', '未知')}）",
        f"- 特殊人群：{profile.get('special_population', '无')}",
    ]
    return "\n".join(lines)


def build_structured_prompt(case: CanonicalCase) -> str:
    prompt = f"""你是一名注重安全性的医疗问答助手。请基于以下病例信息输出结构化结论，严格使用下面 5 个字段：
1. 初步判断
2. 关键依据
3. 风险信号
4. 建议动作
5. 用药/就医注意事项

病例信息：
{render_patient_profile(case.patient_profile)}
- 科室方向：{case.department}
- 主诉：{case.chief_complaint}
- 现病史：{case.history_present_illness}
- 既往史：{case.past_history}

用户问题：
{case.question}

要求：
- 不要给出确定性诊断结论
- 如果存在红旗信号，要明确指出
- 建议动作必须可执行
- 明确说明哪些内容需要线下就医确认
"""
    return textwrap.dedent(prompt).strip()


def build_general_qa_prompt(case: CanonicalCase) -> str:
    return case.question


def build_key_evidence(case: CanonicalCase) -> str:
    evidence_parts = [case.chief_complaint]
    if case.patient_profile.get("age_group") != "未知":
        evidence_parts.append(f"人群特征为{case.patient_profile['age_group']}")
    if case.patient_profile.get("special_population") not in ("无", ""):
        evidence_parts.append(f"属于{case.patient_profile['special_population']}场景")
    first_answer_sentence = split_sentences(case.reference_answer, limit=1)
    if first_answer_sentence:
        evidence_parts.append(f"参考答案提及：{first_answer_sentence[0]}")
    return "；".join(dict.fromkeys(part for part in evidence_parts if part))


def build_risk_signal_text(case: CanonicalCase) -> str:
    if case.safety_flags:
        return "；".join(case.safety_flags) + "。若相关症状持续、加重或出现新的红旗表现，应尽快线下就医。"
    if case.difficulty == "medium":
        return "当前信息未见明确危急红旗，但若症状持续、加重或影响日常功能，需要及时门诊评估。"
    return "当前描述中未见明确危急红旗，但线上建议不能替代面诊，如症状反复或加重需线下就医。"


def build_suggested_action(case: CanonicalCase) -> str:
    if case.difficulty == "high":
        return "优先尽快线下就医；若出现胸痛明显加重、呼吸困难、意识异常或持续高热，建议急诊评估。"
    if case.difficulty == "medium":
        return "建议尽快预约对应科室门诊，结合体格检查或化验结果进一步明确原因。"
    return "可先进行基础观察和对症处理，若 24-48 小时内无改善、症状反复或加重，再安排门诊评估。"


def build_medication_caution(case: CanonicalCase) -> str:
    if "药" in case.reference_answer or "用药" in case.reference_answer:
        return "如需用药，应结合年龄、既往病史和禁忌证，避免自行长期或联合使用处方药。"
    return "不要自行开始抗生素、激素或长期处方药治疗；任何用药需结合线下医生建议。"


def build_structured_answer(case: CanonicalCase) -> str:
    sections = {
        "初步判断": summarize_for_judgement(case.reference_answer, case.department),
        "关键依据": build_key_evidence(case),
        "风险信号": build_risk_signal_text(case),
        "建议动作": build_suggested_action(case),
        "用药/就医注意事项": build_medication_caution(case),
    }
    return "\n".join(f"{name}：{content}" for name, content in sections.items())


def build_hard_negative(case: CanonicalCase) -> str:
    return "\n".join(
        [
            f"初步判断：大概率只是普通的{case.department}小问题，通常不用太担心。",
            f"关键依据：主要根据“{case.chief_complaint}”做出判断。",
            "风险信号：暂时不考虑特殊风险。",
            "建议动作：先自行观察即可，一般不需要尽快线下就医。",
            "用药/就医注意事项：可以按经验自行用药，后续再看情况。",
        ]
    )


def build_bad_negative(case: CanonicalCase) -> str:
    return "\n".join(
        [
            f"初步判断：已经可以确定就是严重的{case.department}疾病。",
            "关键依据：不需要更多依据，症状已经足够说明问题。",
            "风险信号：没有必要额外关注风险信号。",
            "建议动作：无需检查，也不用线下就医，直接按严重疾病处理即可。",
            "用药/就医注意事项：可以立刻自行服用强效处方药，效果通常没问题。",
        ]
    )


def build_preference_row(case: CanonicalCase) -> Dict[str, Any]:
    chosen = build_structured_answer(case)
    negative = build_hard_negative(case) if stable_hash(case.case_id, length=2)[0] in "01234567" else build_bad_negative(case)
    return {
        "case_id": case.case_id,
        "conversations": [{"from": "human", "value": build_structured_prompt(case)}],
        "chosen": chosen,
        "rejected": negative,
        "meta": {
            "department": case.department,
            "difficulty": case.difficulty,
            "safety_flags": case.safety_flags,
        },
    }


def build_sft_row(case: CanonicalCase, track: str) -> Dict[str, Any]:
    if track == "general_qa":
        prompt = build_general_qa_prompt(case)
        answer = case.reference_answer
    else:
        prompt = build_structured_prompt(case)
        answer = build_structured_answer(case)
    return {
        "case_id": case.case_id,
        "track": track,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": answer},
        ],
        "meta": {
            "department": case.department,
            "difficulty": case.difficulty,
            "safety_flags": case.safety_flags,
        },
    }


def build_eval_row(case: CanonicalCase) -> Dict[str, Any]:
    target_answer = build_structured_answer(case)
    gold_fields = parse_structured_answer(target_answer)
    return {
        "case_id": case.case_id,
        "department": case.department,
        "difficulty": case.difficulty,
        "safety_flags": case.safety_flags,
        "prompt": build_structured_prompt(case),
        "target_answer": target_answer,
        "question": case.question,
        "reference_answer": case.reference_answer,
        "gold_fields": gold_fields,
    }


def parse_structured_answer(answer: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for index, (_, title) in enumerate(SECTION_SPECS):
        next_titles = [name for _, name in SECTION_SPECS[index + 1 :]]
        if next_titles:
            stop_pattern = "|".join(re.escape(f"{name}：") for name in next_titles)
            stop_pattern += "|" + "|".join(re.escape(f"{name}:") for name in next_titles)
            pattern = re.compile(
                rf"{re.escape(title)}[：:]\s*(.*?)(?=(?:{stop_pattern})|$)",
                re.S,
            )
        else:
            pattern = re.compile(rf"{re.escape(title)}[：:]\s*(.*)$", re.S)
        match = pattern.search(answer)
        parsed[title] = normalize_text(match.group(1)) if match else ""
    return parsed


def case_to_intermediate_row(case: CanonicalCase) -> Dict[str, Any]:
    row = asdict(case)
    row["patient_profile"] = dict(case.patient_profile)
    row["safety_flags"] = list(case.safety_flags)
    row["coverage_tags"] = list(case.coverage_tags)
    return row


def bucket_report(cases: Sequence[CanonicalCase]) -> Dict[str, Any]:
    return {
        "total_cases": len(cases),
        "by_source": dict(Counter(case.source for case in cases)),
        "by_department": dict(Counter(case.department for case in cases)),
        "by_difficulty": dict(Counter(case.difficulty for case in cases)),
        "flagged_cases": sum(1 for case in cases if case.safety_flags),
    }


def choose_track(case: CanonicalCase, qa_ratio: float) -> str:
    if qa_ratio <= 0:
        return "structured_case"
    threshold = int(qa_ratio * 100)
    bucket = int(stable_hash(case.case_id, length=2), 16) % 100
    return "general_qa" if bucket < threshold else "structured_case"


def build_manual_rubric_csv(path: Path, eval_rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "department",
        "difficulty",
        "question",
        "target_answer",
        "prediction",
        "professional_score",
        "safety_score",
        "consistency_score",
        "actionability_score",
        "overconfidence_score",
        "notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in eval_rows:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "department": row["department"],
                    "difficulty": row["difficulty"],
                    "question": row["question"],
                    "target_answer": row["target_answer"],
                }
            )


def ensure_source_paths(paths: Sequence[str]) -> List[Path]:
    result = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")
        result.append(path)
    return result


def build_bundle(args: argparse.Namespace) -> Dict[str, Any]:
    profile = PROFILE_DEFAULTS[args.profile]
    sft_target = args.sft_target or profile["sft"]
    preference_target = args.preference_target or profile["preference"]
    eval_target = args.eval_target or profile["eval"]

    all_cases: List[CanonicalCase] = []

    for path in ensure_source_paths(args.local_sft_source):
        all_cases.extend(build_cases_from_local_sft_file(path, source_name=path.stem))

    if args.include_hf_public_sources:
        for spec in DEFAULT_HF_REPO_SFT_SOURCES:
            all_cases.extend(
                build_cases_from_hf_repo_file(
                    repo_id=spec["repo_id"],
                    path_in_repo=spec["path"],
                    source_name=spec["source_name"],
                    max_records=args.hf_max_records,
                )
            )

    all_cases = deduplicate_cases(all_cases)
    if not all_cases:
        raise RuntimeError("No valid medical cases were collected. Check the source files or HF dataset access.")

    rng_seed = args.seed
    test_cases = round_robin_sample(all_cases, eval_target, rng_seed)
    remaining = remove_cases(all_cases, test_cases)
    val_target = min(max(args.min_validation_size, int(sft_target * args.validation_ratio)), max(1, len(remaining)))
    val_cases = round_robin_sample(remaining, val_target, rng_seed + 1)
    remaining = remove_cases(remaining, val_cases)
    train_cases = round_robin_sample(remaining, min(sft_target, len(remaining)), rng_seed + 2)

    pref_train_source = list(train_cases)
    pref_val_source = list(val_cases) if val_cases else list(test_cases[: min(32, len(test_cases))])
    pref_train_cases = round_robin_sample(pref_train_source, min(preference_target, len(pref_train_source)), rng_seed + 3)
    pref_val_target = min(max(args.min_preference_validation_size, int(len(pref_train_cases) * args.preference_validation_ratio)), len(pref_val_source))
    pref_val_cases = round_robin_sample(pref_val_source, pref_val_target, rng_seed + 4)

    train_rows = [case_to_intermediate_row(case) for case in train_cases]
    val_rows = [case_to_intermediate_row(case) for case in val_cases]
    test_rows = [case_to_intermediate_row(case) for case in test_cases]

    sft_train_rows = [build_sft_row(case, choose_track(case, args.general_qa_ratio)) for case in train_cases]
    sft_val_rows = [build_sft_row(case, "structured_case") for case in val_cases]
    sft_test_rows = [build_sft_row(case, "structured_case") for case in test_cases]

    reward_train_rows = [build_preference_row(case) for case in pref_train_cases]
    reward_val_rows = [build_preference_row(case) for case in pref_val_cases]

    eval_rows = [build_eval_row(case) for case in test_cases]
    output_root = Path(args.output_root)

    write_jsonl(output_root / "intermediate" / "train" / "train.jsonl", train_rows)
    write_jsonl(output_root / "intermediate" / "val" / "val.jsonl", val_rows)
    write_jsonl(output_root / "intermediate" / "test" / "test.jsonl", test_rows)

    write_jsonl(output_root / "sft" / "train" / "train.jsonl", sft_train_rows)
    write_jsonl(output_root / "sft" / "val" / "val.jsonl", sft_val_rows)
    write_jsonl(output_root / "sft" / "test" / "test.jsonl", sft_test_rows)

    write_jsonl(output_root / "reward" / "train" / "train.jsonl", reward_train_rows)
    write_jsonl(output_root / "reward" / "val" / "val.jsonl", reward_val_rows)

    write_jsonl(output_root / "eval" / "medical_eval_cases.jsonl", eval_rows)
    build_manual_rubric_csv(output_root / "eval" / "manual_review_template.csv", eval_rows)

    report = {
        "profile": args.profile,
        "seed": args.seed,
        "sources": {
            "local_sft_source": list(args.local_sft_source),
            "include_hf_public_sources": args.include_hf_public_sources,
            "hf_max_records": args.hf_max_records,
        },
        "targets": {
            "sft_train": sft_target,
            "preference_train": preference_target,
            "eval_test": eval_target,
        },
        "generated": {
            "intermediate_train": len(train_rows),
            "intermediate_val": len(val_rows),
            "intermediate_test": len(test_rows),
            "sft_train": len(sft_train_rows),
            "reward_train": len(reward_train_rows),
            "reward_val": len(reward_val_rows),
            "eval_test": len(eval_rows),
        },
        "qa_track_ratio": args.general_qa_ratio,
        "train_summary": bucket_report(train_cases),
        "val_summary": bucket_report(val_cases),
        "test_summary": bucket_report(test_cases),
    }
    write_json(output_root / "metadata" / "build_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build structured medical closed-loop datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build the dataset bundle.")
    build_parser.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS), default="smoke")
    build_parser.add_argument("--output-root", default="data/medical_closed_loop/smoke")
    build_parser.add_argument(
        "--local-sft-source",
        nargs="+",
        default=DEFAULT_LOCAL_SFT_SOURCES,
        help="Local ShareGPT or QA JSONL files used as the raw candidate pool.",
    )
    build_parser.add_argument(
        "--include-hf-public-sources",
        action="store_true",
        help="Also pull candidate cases from the default public HF datasets.",
    )
    build_parser.add_argument("--hf-max-records", type=int, default=-1)
    build_parser.add_argument("--sft-target", type=int, default=0)
    build_parser.add_argument("--preference-target", type=int, default=0)
    build_parser.add_argument("--eval-target", type=int, default=0)
    build_parser.add_argument("--general-qa-ratio", type=float, default=0.2)
    build_parser.add_argument("--validation-ratio", type=float, default=0.1)
    build_parser.add_argument("--preference-validation-ratio", type=float, default=0.1)
    build_parser.add_argument("--min-validation-size", type=int, default=64)
    build_parser.add_argument("--min-preference-validation-size", type=int, default=32)
    build_parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "build":
        report = build_bundle(args)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
