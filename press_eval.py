from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()  # this will read .env and set the env vars

# --------- Domain prompts template for the model --------- #

def load_prompt(path: str | Path) -> str:
    """Load the PRESS prompt from an external text file."""
    path = Path(path)
    return path.read_text(encoding="utf-8")

trns_prompt = load_prompt("/Users/mohamad22/Desktop/PRESS_eval/prompts/1_trns_prompt.txt")
bool_prompt = load_prompt("/Users/mohamad22/Desktop/PRESS_eval/prompts/2_bool_prompt.txt")
subj_prompt = load_prompt("/Users/mohamad22/Desktop/PRESS_eval/prompts/3_subj_prompt.txt")
text_prompt = load_prompt("/Users/mohamad22/Desktop/PRESS_eval/prompts/4_text_prompt.txt")
synt_prompt = load_prompt("/Users/mohamad22/Desktop/PRESS_eval/prompts/5_synt_prompt.txt")
limt_prompt = load_prompt("/Users/mohamad22/Desktop/PRESS_eval/prompts/6_limt_prompt.txt")

# --------- Data structures --------- #

@dataclass
class SystematicReviewInfo:
    """Holds the raw JSON metadata about the systematic review (PICO, objective, etc.)."""
    data: Dict[str, Any]


@dataclass
class SearchStrategy:
    """Holds the search strategy text for a given database (e.g., PubMed)."""
    database: str
    text: str

@dataclass
class PressDomainQuestionResult:
    """Boolean answer to a single checklist question within a PRESS domain."""
    question: str   # exact question text
    answer: bool    # True or False


@dataclass
class PressDomainResult:
    """
    Result for a single PRESS 2015 domain
    (e.g., Boolean/proximity operators, subject headings).
    """
    domain: str                               # e.g., "boolean_proximity"
    questions: List[PressDomainQuestionResult]
    justification: str                        # one narrative paragraph for the domain

# --------- Core stub  --------- #

def review_strategy_with_press(
    review_info: SystematicReviewInfo,
    strategy: SearchStrategy,
    system_prompt: str,
    model: str = "gpt-4o-mini",
) -> PressDomainResult:
    """
    Generic helper to review a single PRESS 2015 domain.

    - You pass in the domain-specific system prompt (e.g. trns_prompt, bool_prompt, ...).
    - Builds chat messages from the systematic review info and search strategy.
    - Expects the model to return a JSON object with keys:
        - "domain": string
        - "questions": list of { "question": string, "answer": boolean }
        - "justification": string
    - Requires OPENAI_API_KEY to be set in the environment, unless you pass a key
      when constructing the OpenAI client yourself.
    """
    client = OpenAI()  # reads OPENAI_API_KEY from environment by default

    messages = build_press_messages(review_info, strategy, system_prompt)

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )

    content = completion.choices[0].message.content or ""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Model did not return valid JSON. Raw content was:\n"
            f"{content}"
        ) from exc

    return _dict_to_press_domain_result(data)


# --------- File loading helpers --------- #

def load_systematic_review_info(path: Path) -> SystematicReviewInfo:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return SystematicReviewInfo(data=data)


def load_search_strategy(path: Path, database_name: str = "PubMed") -> SearchStrategy:
    with path.open("r", encoding="utf-8") as f:
        text = f.read()
    return SearchStrategy(database=database_name, text=text)

def build_press_messages(
    review_info: SystematicReviewInfo,
    strategy: SearchStrategy,
    system_prompt: str,
) -> list[dict[str, str]]:
    """
    Build the chat-style messages to send to the LLM:
    - a system message containing the PRESS instructions,
    - a user message containing the review info and the search strategy.
    """
    review_info_pretty = json.dumps(review_info.data, indent=2, ensure_ascii=False)

    user_content = (
        "You are reviewing the following systematic review and search strategy.\n\n"
        "SYSTEMATIC REVIEW INFO (JSON):\n"
        f"{review_info_pretty}\n\n"
        f"SEARCH STRATEGY (database={strategy.database}):\n"
        f"{strategy.text}\n"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

def _dict_to_press_domain_result(data: Dict[str, Any]) -> PressDomainResult:
    """Convert a JSON-like dict from the model into a PressDomainResult."""
    questions_data = data.get("questions", []) or []
    questions: List[PressDomainQuestionResult] = [
        PressDomainQuestionResult(
            question=q["question"],
            answer=bool(q["answer"]),
        )
        for q in questions_data
    ]

    return PressDomainResult(
        domain=data["domain"],
        questions=questions,
        justification=data.get("justification", ""),
    )


def press_domain_to_dict(result: PressDomainResult) -> Dict[str, Any]:
    """Convert a PressDomainResult into a plain JSON-serializable dict."""
    return {
        "domain": result.domain,
        "questions": [
            {
                "question": q.question,
                "answer": q.answer,
            }
            for q in result.questions
        ],
        "justification": result.justification,
    }

# --------- demo --------- #

def main() -> None:
    ss_info_path = Path("vitd_falls_ss_info.json")
    ss_strategy_path = Path("vitd_falls_pubmed_ss.txt")

    review_info = load_systematic_review_info(ss_info_path)
    search_strategy = load_search_strategy(ss_strategy_path, database_name="PubMed")

    try:
        # Evaluate only the first domain: translation of the research question
        trns_result = review_strategy_with_press(
            review_info=review_info,
            strategy=search_strategy,
            system_prompt=trns_prompt,
        )
    except Exception as e:
        print("\n[ERROR] Translation domain PRESS review failed:")
        print(e)
    else:
        trns_dict = press_domain_to_dict(trns_result)

        output_path = Path("translation_output.json")
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(trns_dict, f, indent=2, ensure_ascii=False)

        print(f"\nSaved translation PRESS review to: {output_path}")


if __name__ == "__main__":
    main()