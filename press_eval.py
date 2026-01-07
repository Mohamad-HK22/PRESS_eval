from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()  # this will read .env and set the env vars

# --------- System prompt template for the model --------- #

def load_press_system_prompt(path: Path = Path("press_system_prompt.txt")) -> str:
    """Load the PRESS system prompt from an external text file."""
    return path.read_text(encoding="utf-8")

PRESS_SYSTEM_PROMPT = load_press_system_prompt()

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
class PressDomainResult:
    """
    Result for a single PRESS 2015 domain
    (e.g., Boolean/proximity operators, subject headings).
    """
    domain: str              # machine-friendly name of the domain
    passed: bool             # True if this domain is judged adequate
    issues: List[str]        # short descriptions of problems found
    severity: str            # e.g., "none", "minor", "major"


@dataclass
class PressReviewResult:
    """
    Overall PRESS review result for one database search strategy.
    """
    strategy_database: str             # e.g., "PubMed"
    overall_pass: bool                 # overall adequacy of the strategy
    overall_reason: str                # short narrative justification
    domains: List[PressDomainResult]   # per-domain results


# --------- Core stub  --------- #

def review_strategy_with_press(
    review_info: SystematicReviewInfo,
    strategy: SearchStrategy,
    model: str = "gpt-4o-mini",
) -> PressReviewResult:
    """
    Call the OpenAI Chat Completions API to perform a PRESS 2015 review.

    - Uses the global PRESS_SYSTEM_PROMPT (loaded from the external file).
    - Builds chat messages from the systematic review info and search strategy.
    - Expects the model to return a single JSON object matching the PressReviewResult schema.
    - Requires OPENAI_API_KEY to be set in the environment, unless you pass a key
      when constructing the OpenAI client yourself.
    """
    client = OpenAI()  # reads OPENAI_API_KEY from environment by default

    messages = build_press_messages(review_info, strategy, PRESS_SYSTEM_PROMPT)

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )

    content = completion.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Model did not return valid JSON. Raw content was:\n"
            f"{content}"
        ) from exc

    return _dict_to_press_review_result(data)


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

def _dict_to_press_review_result(data: Dict[str, Any]) -> "PressReviewResult":
    """Convert a JSON-like dict from the model into PressReviewResult objects."""
    domain_dicts = data.get("domains", []) or []
    domains: List[PressDomainResult] = [
        PressDomainResult(
            domain=d["domain"],
            passed=bool(d["passed"]),
            issues=list(d.get("issues", [])),
            severity=d["severity"],
        )
        for d in domain_dicts
    ]

    return PressReviewResult(
        strategy_database=data["strategy_database"],
        overall_pass=bool(data["overall_pass"]),
        overall_reason=data["overall_reason"],
        domains=domains,
    )

def press_review_to_dict(result: PressReviewResult) -> Dict[str, Any]:
    """Convert a PressReviewResult into a plain JSON-serializable dict."""
    return {
        "strategy_database": result.strategy_database,
        "overall_pass": result.overall_pass,
        "overall_reason": result.overall_reason,
        "domains": [
            {
                "domain": d.domain,
                "passed": d.passed,
                "issues": "; ".join(d.issues),
                "severity": d.severity,
            }
            for d in result.domains
        ],
    }

def press_review_to_dict(result: PressReviewResult) -> Dict[str, Any]:
    """Convert a PressReviewResult into a plain JSON-serializable dict."""
    return {
        "strategy_database": result.strategy_database,
        "overall_pass": result.overall_pass,
        "overall_reason": result.overall_reason,
        "domains": [
            {
                "domain": d.domain,
                "passed": d.passed,
                "issues": "; ".join(d.issues),
                "severity": d.severity,
            }
            for d in result.domains
        ],
    }

# --------- demo --------- #

def main() -> None:
    ss_info_path = Path("vitd_falls_ss_info.json")
    ss_strategy_path = Path("vitd_falls_pubmed_ss.txt")

    review_info = load_systematic_review_info(ss_info_path)
    search_strategy = load_search_strategy(ss_strategy_path, database_name="PubMed")

    # print("=== Systematic Review Info (JSON) ===")
    # pprint(review_info.data)

    # print("\n=== Search Strategy (PubMed) ===")
    # print(search_strategy.text)

    # # Build example messages for the model
    # messages = build_press_messages(review_info, search_strategy, PRESS_SYSTEM_PROMPT)

    # print("\n=== Example user message to send to the model ===")
    # print(messages[1]["content"])

    try:
        press_result = review_strategy_with_press(review_info, search_strategy)
    except Exception as e:
        print("\n[ERROR] PRESS review failed:")
        print(e)
    else:
        review_dict = press_review_to_dict(press_result)

        output_path = Path("output.json")

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(review_dict, f, indent=2, ensure_ascii=False)

        print(f"\nSaved PRESS review to: {output_path}")


if __name__ == "__main__":
    main()