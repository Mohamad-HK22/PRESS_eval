from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Union

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

    @classmethod
    def from_json_file(cls, path: Path) -> "SystematicReviewInfo":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data=data)

    # ---- Convenience accessors (read-only) ----
    @property
    def search_data(self) -> Dict[str, Any]:
        return self.data["search_data"]

    @property
    def strategy(self) -> Dict[str, Any]:
        return self.search_data["search_strategy_data"]

    @property
    def min_year(self) -> Optional[int]:
        return self.search_data.get("min_year")  # safe if missing

    @property
    def max_year(self) -> Optional[int]:
        return self.search_data.get("max_year")

    @property
    def concepts(self) -> List[Any]:
        return self.strategy.get("concepts", [])

    @property
    def connected_concepts(self) -> List[Any]:
        return self.strategy.get("connected_concepts", [])

    @property
    def connected_keywords(self) -> List[Any]:
        return self.strategy.get("connected_keywords", [])

    @property
    def objective(self) -> Optional[str]:
        return self.strategy.get("objective")

    @property
    def pico(self) -> Dict[str, Any]:
        return self.strategy.get("pico_elements", {})
    
    def __str__(self) -> str:
        return str(self.data)


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


JSONType = Union[dict, list, str, int, float, bool, None]

def combine_to_json_file(parts: Dict[str, JSONType], out_path: str = "PRESS_review.json", *, indent: int = 2) -> Path:
    """
    parts example:
      {
        "pico": {...},
        "objective": "....",
        "min_year": 2010
      }
    Saves them into a single JSON file and returns the file path.
    """
    out_file = Path(out_path)
    out_file.write_text(
        json.dumps(parts, indent=indent, ensure_ascii=False),
        encoding="utf-8"
    )
    return out_file

# --------- demo --------- #

def main() -> None:
    ss_info_path = Path("vitd_falls_ss_info.json")
    ss_strategy_path = Path("vitd_falls_pubmed_ss.txt")

    review_info = load_systematic_review_info(ss_info_path)
    search_strategy = load_search_strategy(ss_strategy_path, database_name="PubMed")

    try:
        # Evaluate domain by domain
        trns_result = review_strategy_with_press(
            review_info=SystematicReviewInfo(data={"objective":review_info.objective, "pico":review_info.pico, "concepts":review_info.concepts, "connected_concepts":review_info.connected_concepts}),
            strategy=search_strategy,
            system_prompt=trns_prompt,
        )
        bool_result = review_strategy_with_press(
            review_info=SystematicReviewInfo(data={"objective": review_info.objective, "pico": review_info.pico}),
            strategy=search_strategy,
            system_prompt=bool_prompt,
        )
        subj_result = review_strategy_with_press(
            review_info=SystematicReviewInfo(data={"objective":review_info.objective, "pico":review_info.pico, "concepts":review_info.concepts}),
            strategy=search_strategy,
            system_prompt=subj_prompt,
        )
        text_result = review_strategy_with_press(
            review_info=SystematicReviewInfo(data={"objective":review_info.objective, "pico":review_info.pico, "concepts":review_info.concepts, "connected_keywords":review_info.connected_keywords}),
            strategy=search_strategy,
            system_prompt=text_prompt,
        )
        synt_result = review_strategy_with_press(
            review_info=SystematicReviewInfo(data={"concepts":review_info.concepts, "connected_concepts":review_info.connected_concepts}),
            strategy=search_strategy,
            system_prompt=synt_prompt,
        )
        limt_result = review_strategy_with_press(
            review_info=review_info,
            strategy=search_strategy,
            system_prompt=limt_prompt,
        )
    except Exception as e:
        print("\n[ERROR] PRESS review failed:")
        print(e)
    else:
        trns_dict = press_domain_to_dict(trns_result)
        bool_dict = press_domain_to_dict(bool_result)
        subj_dict = press_domain_to_dict(subj_result)
        text_dict = press_domain_to_dict(text_result)
        synt_dict = press_domain_to_dict(synt_result)
        limt_dict = press_domain_to_dict(limt_result)

        # Combine all domain results into one JSON structure
        combined_result = {
            "strategy_database": search_strategy.database,
            "domains": [
                trns_dict,
                bool_dict,
                subj_dict,
                text_dict,
                synt_dict,
                limt_dict,
            ],
        }

        output_path = combine_to_json_file(combined_result, "press_review_output3.json")
        print(f"\nSaved PRESS review to: {output_path}")

if __name__ == "__main__":
    main()