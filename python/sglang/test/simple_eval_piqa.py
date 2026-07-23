# PIQA: Physical Interaction Question Answering
# Dataset format supported here:
#   JSONL/CSV/JSON records with fields: goal, sol1, sol2, label.
# The original PIQA release may store labels in a separate *-labels.lst file;
# passing the dataset directory is supported for that layout.

import csv
import json
import re
from pathlib import Path
from typing import Optional

from sglang.test import simple_eval_common as common
from sglang.test.simple_eval_common import (
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
)


QUERY_TEMPLATE_PIQA = """
Choose the more plausible solution for the physical interaction problem.
The last line of your response should be exactly: 'Answer: $LETTER'
where LETTER is one of A or B.

Goal: {goal}

A) {sol1}
B) {sol2}
""".strip()

ANSWER_PATTERN_PIQA = r"(?i)Answer\s*:\s*([AB])"


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "examples", "validation", "train", "test"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported PIQA JSON structure: {path}")


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_labels(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _find_piqa_files(path: Path) -> tuple[Path, Optional[Path]]:
    if path.is_file():
        return path, None

    data_candidates = [
        "validation.jsonl",
        "valid.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "train.jsonl",
        "piqa.jsonl",
        "data.jsonl",
        "validation.csv",
        "valid.csv",
        "dev.csv",
        "test.csv",
        "train.csv",
        "piqa.csv",
        "data.csv",
    ]
    label_candidates = [
        "validation-labels.lst",
        "valid-labels.lst",
        "dev-labels.lst",
        "test-labels.lst",
        "train-labels.lst",
        "labels.lst",
    ]

    data_path = next((path / name for name in data_candidates if (path / name).exists()), None)
    if data_path is None:
        raise FileNotFoundError(
            f"No PIQA data file found in {path}. Expected JSONL/CSV with fields "
            "goal, sol1, sol2, label."
        )

    label_path = next((path / name for name in label_candidates if (path / name).exists()), None)
    return data_path, label_path


def _load_piqa_examples(path: str) -> list[dict]:
    data_path, label_path = _find_piqa_files(Path(path).expanduser())
    suffix = data_path.suffix.lower()
    if suffix == ".jsonl":
        examples = _read_jsonl(data_path)
    elif suffix == ".json":
        examples = _read_json(data_path)
    elif suffix == ".csv":
        examples = _read_csv(data_path)
    else:
        raise ValueError(f"Unsupported PIQA data file type: {data_path}")

    if label_path is not None and examples and "label" not in examples[0]:
        labels = _read_labels(label_path)
        if len(labels) != len(examples):
            raise ValueError(
                f"PIQA label count mismatch: {len(labels)} labels for {len(examples)} examples"
            )
        examples = [example | {"label": label} for example, label in zip(examples, labels)]

    required = ("goal", "sol1", "sol2", "label")
    missing = [field for field in required if examples and field not in examples[0]]
    if missing:
        raise ValueError(f"PIQA data missing required fields {missing}: {data_path}")

    return examples


def _normalize_label(label) -> str:
    value = str(label).strip().lower()
    if value in ("0", "a", "sol1", "solution1"):
        return "A"
    if value in ("1", "b", "sol2", "solution2"):
        return "B"
    raise ValueError(f"Unsupported PIQA label: {label!r}")


def _extract_answer(response_text: str) -> Optional[str]:
    match = re.search(ANSWER_PATTERN_PIQA, response_text or "")
    if match:
        return match.group(1).upper()

    # Fallback for terse models that only return "A" or "B".
    match = re.search(r"(?im)^\s*([AB])[\).]?\s*$", response_text or "")
    if match:
        return match.group(1).upper()
    return None


class PIQAEval(Eval):
    def __init__(
        self,
        filename: str,
        num_examples: Optional[int],
        num_threads: int,
    ):
        examples = _load_piqa_examples(filename)
        if num_examples:
            examples = examples[:num_examples]
        self.examples = examples
        self.num_threads = num_threads

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            correct_answer = _normalize_label(row["label"])
            prompt = QUERY_TEMPLATE_PIQA.format(
                goal=row["goal"],
                sol1=row["sol1"],
                sol2=row["sol2"],
            )
            prompt_messages = [sampler._pack_message(content=prompt, role="user")]
            response_text = sampler(prompt_messages) or ""
            extracted_answer = _extract_answer(response_text)
            score = 1.0 if extracted_answer == correct_answer else 0.0
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=correct_answer,
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics={
                    "chars": len(response_text),
                    "invalid": 1.0 if extracted_answer is None else 0.0,
                },
            )

        results = common.map_with_progress(fn, self.examples, self.num_threads)
        return common.aggregate_results(results)
