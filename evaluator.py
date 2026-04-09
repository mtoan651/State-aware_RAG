"""
evaluator.py – LLM-as-Judge benchmark for the RAG agent.

Usage
-----
# Run in GRAPH mode (default)
python evaluator.py

# Run in LEGACY (ReAct) mode
TEST_MODE=LEGACY python evaluator.py

# Compare both embedding models side-by-side
python evaluator.py --compare-models

# Change agent mode on the command line
python evaluator.py --mode LEGACY
"""

import sys
import os
import datetime
import warnings
import re
import argparse
import time

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")
warnings.filterwarnings("ignore", message=".*Convert_system_message_to_human.*")
warnings.filterwarnings("ignore", message=".*API key must be provided.*")

from termcolor import colored
from langchain_core.prompts import ChatPromptTemplate

from config import get_llm, ACTIVE_EMBEDDING, CHUNK_SIZE


# ---------------------------------------------------------------------------
# Settings (overridable by env vars or CLI args)
# ---------------------------------------------------------------------------
TEST_MODE = os.getenv("TEST_MODE", "GRAPH").upper()   # "GRAPH" | "LEGACY"


# ---------------------------------------------------------------------------
# Dual logger: writes to console AND a log file simultaneously
# ---------------------------------------------------------------------------
class DualLogger:
    _ansi = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    def __init__(self, filename: str = "evaluation_log.txt"):
        self.terminal = sys.stdout
        self.log      = open(filename, "w", encoding="utf-8")

    def isatty(self) -> bool:
        return self.terminal.isatty()

    @property
    def encoding(self) -> str:
        return self.terminal.encoding

    def fileno(self) -> int:
        return self.terminal.fileno()

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(self._ansi.sub("", message))
        self.log.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------
def grade_answer_with_llm(
    question: str,
    agent_answer: str,
    expected_facts: list[str],
    forbidden_facts: list[str],
) -> str:
    llm = get_llm(temperature=0)

    prompt = ChatPromptTemplate.from_template("""
You are a strict grading assistant. Evaluate whether AGENT_ANSWER satisfies the criteria below.

QUESTION: {question}
AGENT_ANSWER: {agent_answer}

CRITERIA 1 — Must Include: The answer MUST semantically contain these facts: {expected_facts}
CRITERIA 2 — Forbidden:   The answer MUST NOT mention: {forbidden_facts}

Grading notes:
- "391 billion" and "391,000 million" are equivalent → PASS.
- Correct facts expressed in a different language → PASS.
- Any hallucinated data or mention of a forbidden topic → FAIL.
- If the question is a trap (asks for info not in documents), the agent should say
  "I don't know" or similar → that is a PASS for honesty.

OUTPUT EXACTLY ONE WORD: PASS or FAIL.
""")

    chain  = prompt | llm
    result = chain.invoke({
        "question":       question,
        "agent_answer":   agent_answer,
        "expected_facts": str(expected_facts),
        "forbidden_facts": str(forbidden_facts),
    })
    return result.content.strip().upper()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "name":         "Test A: Apple Revenue (Chinese)",
        "question":     "Apple 2024 年的總營收 (Total net sales) 是多少？",
        "must_contain": ["391", "billion"],
        "forbidden":    ["Tesla"],
    },
    {
        "name":         "Test B: Tesla R&D (Chinese)",
        "question":     "Tesla 2024 年的研發費用 (R&D expenses) 是多少？",
        "must_contain": ["4.77", "billion"],
        "forbidden":    ["Apple"],
    },
    {
        "name":         "Test D: Apple Services Cost (Chinese)",
        "question":     "Apple 2024 年的「服務成本 (Cost of sales - Services)」是多少？",
        "must_contain": ["25", "billion", "25,119"],
        "forbidden":    [],
    },
    {
        "name":         "Test E: Tesla Energy Revenue (Chinese)",
        "question":     "Tesla 2024 年的「能源發電與儲存 (Energy generation and storage)」營收是多少？",
        "must_contain": ["23.7", "billion", "23,767"],
        "forbidden":    [],
    },
    {
        "name":         "Test G: Unknown Info (Trap)",
        "question":     "Apple 計畫在 2025 年發布的 iPhone 17 預計售價是多少？",
        "must_contain": ["unknown", "provide", "mention", "does not", "無法", "未提及"],
        "forbidden":    ["1000", "999", "1200"],
    },
    {
        "name":         "Test A1: Apple Revenue (English)",
        "question":     "What was Apple's Total Net Sales for the fiscal year 2024?",
        "must_contain": ["391", "billion", "391,035"],
        "forbidden":    ["Tesla"],
    },
    {
        "name":         "Test A2: Tesla Automotive Revenue (English)",
        "question":     "What is the specific revenue figure for 'Automotive sales' for Tesla in 2024?",
        "must_contain": ["78", "billion", "78,512"],
        "forbidden":    ["Apple"],
    },
    {
        "name":         "Test B1: Apple R&D (Mixed)",
        "question":     "Apple 2024 年的研發費用 (Research and development expenses) 是多少？",
        "must_contain": ["31", "billion", "31,370"],
        "forbidden":    ["Tesla"],
    },
    {
        "name":         "Test B2: Tesla CapEx (Mixed)",
        "question":     "Tesla 在 2024 年的資本支出 (Capital Expenditures) 是多少？",
        "must_contain": ["11", "billion", "11,153"],
        "forbidden":    ["Apple"],
    },
    {
        "name":         "Test C1: R&D Comparison (English)",
        "question":     "Compare the R&D expenses of Apple and Tesla in 2024. Who spent more?",
        "must_contain": ["Apple", "Apple spent more"],
        "forbidden":    [],
    },
    {
        "name":         "Test C2: Gross Margin Analysis (English)",
        "question":     "Which company had a higher Total Gross Margin percentage in 2024, Apple or Tesla? Provide approximate percentages.",
        "must_contain": ["Apple", "Tesla", "46", "18", "Apple"],
        "forbidden":    [],
    },
    {
        "name":         "Test D1: Apple Services Cost (English)",
        "question":     "Per the Consolidated Statements of Operations, what was Apple's 'Cost of sales' for 'Services' in 2024?",
        "must_contain": ["25", "billion", "25,119"],
        "forbidden":    [],
    },
    {
        "name":         "Test E1: 2025 Projection Trap (Mixed)",
        "question":     "財報中有提到 Apple 2025 年預計的 iPhone 銷量目標嗎？",
        "must_contain": ["no", "not mentioned", "does not provide", "沒有提到", "未知"],
        "forbidden":    ["100 million", "200 million", "increase"],
    },
    {
        "name":         "Test F1: CEO Identity (English)",
        "question":     "Who signed the 10-K report as the Chief Executive Officer for Tesla?",
        "must_contain": ["Elon Musk"],
        "forbidden":    ["Tim Cook"],
    },
]


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------
def run_evaluation(mode: str = TEST_MODE) -> tuple[int, int]:
    """Run all test cases and return (score, total)."""
    from langgraph_agent import run_graph_agent, run_legacy_agent

    score = 0
    total = len(TEST_CASES)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*56}")
    print(f"  ASSIGNMENT 3 EVALUATION REPORT")
    print(f"  Time:            {timestamp}")
    print(f"  Agent Mode:      {mode}")
    print(f"  Embedding Model: {ACTIVE_EMBEDDING}")
    print(f"  Chunk Size:      {CHUNK_SIZE}")
    print(f"{'='*56}\n")
    print(colored("🚀 STARTING EVALUATION …\n", "cyan", attrs=["bold"]))

    for test in TEST_CASES:
        print(f"▶ {test['name']}")
        start = time.time()
        try:
            if mode == "GRAPH":
                answer = run_graph_agent(test["question"])
            else:
                answer = run_legacy_agent(test["question"])

            # Strip any trailing Observation artefacts
            clean = answer.split("Observation:")[0].strip()
            display = (clean[:300] + "…") if len(clean) > 300 else clean

            verdict = grade_answer_with_llm(
                test["question"],
                clean,
                test["must_contain"],
                test["forbidden"],
            )

            elapsed = time.time() - start
            print(f"  A: {display}")
            if "PASS" in verdict:
                score += 1
                print(colored(f"  ✅ PASS  ({elapsed:.2f}s)", "green"))
            else:
                print(colored(f"  ❌ FAIL  ({elapsed:.2f}s)", "red"))
                print(f"     Expected: {test['must_contain']}")
                print(f"     Forbidden: {test['forbidden']}")

        except Exception as exc:
            print(colored(f"  💥 CRASH: {exc}", "red"))

        print("-" * 50)

    pct = score / total * 100
    print(colored(f"\n📊 FINAL SCORE: {score}/{total}  ({pct:.1f}%)", "magenta", attrs=["bold"]))
    return score, total


# ---------------------------------------------------------------------------
# Model comparison helper
# ---------------------------------------------------------------------------
def compare_models(mode: str = TEST_MODE) -> None:
    """Run the evaluation for each available embedding model and print a summary."""
    import importlib
    import config
    from config import EMBEDDING_MODELS

    results: dict[str, tuple[int, int]] = {}

    for model_key in EMBEDDING_MODELS:
        print(colored(f"\n{'#'*56}", "cyan"))
        print(colored(f"  EMBEDDING MODEL: {model_key}", "cyan", attrs=["bold"]))
        print(colored(f"{'#'*56}", "cyan"))

        # Patch the active embedding at runtime and reload retrievers
        os.environ["EMBEDDING_MODEL"] = model_key
        config.ACTIVE_EMBEDDING = model_key

        import langgraph_agent
        importlib.reload(langgraph_agent)

        # Re-import the agent functions after reload
        from langgraph_agent import run_graph_agent, run_legacy_agent  # noqa: F811

        score, total = run_evaluation(mode)
        results[model_key] = (score, total)

    # Summary table
    print(colored("\n" + "="*56, "magenta"))
    print(colored("  EMBEDDING MODEL COMPARISON SUMMARY", "magenta", attrs=["bold"]))
    print(colored("="*56, "magenta"))
    print(f"  {'Model':<10}  {'Score':>8}  {'%':>6}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*6}")
    for key, (s, t) in results.items():
        print(f"  {key:<10}  {s}/{t:>4}    {s/t*100:>5.1f}%")
    print(colored("="*56, "magenta"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG agent evaluation.")
    parser.add_argument(
        "--mode", default=TEST_MODE, choices=["GRAPH", "LEGACY"],
        help="Agent mode to evaluate",
    )
    parser.add_argument(
        "--compare-models", action="store_true",
        help="Run evaluation for all embedding models and print a comparison",
    )
    parser.add_argument(
        "--debug-chunks", action="store_true",
        help="Print full retrieved chunk content during retrieval",
    )
    parser.add_argument(
        "--rerank", action="store_true",
        help="Re-rank retrieved chunks with a cross-encoder before generation",
    )
    args = parser.parse_args()

    import langgraph_agent
    if args.debug_chunks:
        langgraph_agent.DEBUG_CHUNKS = True
    if args.rerank:
        langgraph_agent.USE_RERANKER = True

    log_filename = (
        f"evaluation_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    )
    sys.stdout = DualLogger(log_filename)

    if args.compare_models:
        compare_models(args.mode)
    else:
        run_evaluation(args.mode)

    sys.stdout = sys.stdout.terminal  # type: ignore[union-attr]
    print(f"\n[System] Log saved to {log_filename}")
