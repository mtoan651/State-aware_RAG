"""
langgraph_agent.py – State-aware RAG agent with LangGraph + ReAct fallback.

Tasks implemented
-----------------
A  run_legacy_agent  – ReAct prompt with English-only / year-precision / honesty constraints
B  retrieve_node     – LLM-based intelligent router → ["apple", "tesla", "both", "none"]
C  grade_documents_node – Binary relevance grader → "yes" / "no"
D  rewrite_node      – Vague-query → precise financial terminology rewriter
E  generate_node     – Context-grounded answer generator with source citations
"""

import os
import json
from typing import TypedDict
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_chroma import Chroma
from termcolor import colored
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import get_embeddings, get_llm, get_db_folder, FILES, ACTIVE_EMBEDDING

DEBUG_CHUNKS = False  # set via evaluator.py --debug-chunks
USE_RERANKER = True  # set via evaluator.py --rerank

_reranker = None  # lazy-loaded cross-encoder


def get_reranker():
    """Lazy-load the cross-encoder re-ranker (only when USE_RERANKER=True)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print(colored("🔄 Loading re-ranker: cross-encoder/ms-marco-MiniLM-L-6-v2", "cyan"))
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


# ---------------------------------------------------------------------------
# Retry decorator – provider-agnostic, 3 attempts with exponential back-off
# ---------------------------------------------------------------------------
retry_logic = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=5, max=30),
    retry=retry_if_exception_type(Exception),
)


# ---------------------------------------------------------------------------
# Vector DB initialisation
# ---------------------------------------------------------------------------
def initialize_vector_dbs(model_key: str = ACTIVE_EMBEDDING) -> dict:
    """Load ChromaDB retrievers for all known documents."""
    embeddings = get_embeddings(model_key)
    db_root    = get_db_folder(model_key)
    retrievers: dict = {}

    for key in FILES:
        persist_dir = os.path.join(db_root, key)
        if os.path.exists(persist_dir):
            vs = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            retrievers[key] = vs.as_retriever(search_kwargs={"k": 5})
        else:
            print(colored(
                f"❌ DB for '{key}' not found at {persist_dir}. "
                f"Run: python build_rag.py",
                "red",
            ))

    return retrievers


RETRIEVERS = initialize_vector_dbs()


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question:      str
    documents:     str
    generation:    str
    search_count:  int
    needs_rewrite: str   # "yes" = relevant, "no" = irrelevant → trigger rewrite


# ---------------------------------------------------------------------------
# Task B – Intelligent Router
# ---------------------------------------------------------------------------
@retry_logic
def retrieve_node(state: AgentState) -> dict:
    print(colored("--- 🔍 RETRIEVE ---", "blue"))
    question = state["question"]
    llm = get_llm()

    # ── Task B prompt ──────────────────────────────────────────────────────
    router_prompt = f"""You are a query router for a financial document system.

Classify the user question into EXACTLY ONE of these four categories:
  "apple"  – question is about Apple Inc. only
             (keywords: Apple, AAPL, iPhone, Mac, iPad, Tim Cook, iOS, App Store)
  "tesla"  – question is about Tesla Inc. only
             (keywords: Tesla, TSLA, EV, electric vehicle, Elon Musk, Cybertruck, Powerwall)
  "both"   – question asks to compare or combine data from BOTH companies
             (keywords: compare, which company, both, versus, vs.)
  "none"   – question is unrelated to Apple or Tesla

Rules:
- Output ONLY valid JSON with a single key "datasource".
- Do not include any explanation or markdown.

Example outputs:
  {{"datasource": "apple"}}
  {{"datasource": "tesla"}}
  {{"datasource": "both"}}
  {{"datasource": "none"}}

User Question: {question}
"""
    # ──────────────────────────────────────────────────────────────────────

    try:
        response = llm.invoke(router_prompt)
        content  = response.content.strip()
        # Strip optional markdown code fences
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        res_json = json.loads(content)
        target   = res_json.get("datasource", "both")
    except Exception as e:
        print(colored(f"⚠️  Router parse error: {e}. Defaulting to 'both'.", "yellow"))
        target = "both"

    print(colored(f"🎯 Routed to: {target}", "cyan"))

    # Resolve which retrievers to query
    if target == "both":
        targets = list(FILES.keys())
    elif target in FILES:
        targets = [target]
    else:
        targets = []  # "none" or unknown → empty context

    docs_content = ""
    for t in targets:
        if t in RETRIEVERS:
            docs = RETRIEVERS[t].invoke(question)

            if USE_RERANKER and docs:
                reranker = get_reranker()
                pairs  = [(question, doc.page_content) for doc in docs]
                scores = reranker.predict(pairs)
                docs   = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
                print(colored(f"   [Reranker] scores: {[round(float(s), 3) for s in sorted(scores, reverse=True)]}", "magenta"))

            label = t.capitalize()
            print(colored(f"   [{label}] Retrieved {len(docs)} chunks:", "blue"))
            if DEBUG_CHUNKS:
                for i, doc in enumerate(docs):
                    meta = doc.metadata
                    page = meta.get("page", "?")
                    source = meta.get("source", "")
                    print(colored(f"     ── chunk[{i}] page={page} | src={source} ──", "dark_grey"))
                    print(colored(doc.page_content, "dark_grey"))
                    print()
            docs_content += f"\n\n[Source: {label} 10-K]\n" + "\n".join(
                d.page_content for d in docs
            )

    return {
        "documents":    docs_content,
        "search_count": state["search_count"] + 1,
    }


# ---------------------------------------------------------------------------
# Task C – Relevance Grader
# ---------------------------------------------------------------------------
@retry_logic
def grade_documents_node(state: AgentState) -> dict:
    print(colored("--- ⚖️  GRADE ---", "yellow"))
    question  = state["question"]
    documents = state["documents"]
    llm = get_llm()

    # ── Task C prompt ──────────────────────────────────────────────────────
    system_msg = SystemMessage(content=(
        "You are a relevance grader for a financial RAG system.\n"
        "Assess whether the retrieved document contains information that can help "
        "answer the user's question.\n\n"
        "Output rules:\n"
        "  - If the document IS relevant → respond with exactly: yes\n"
        "  - If the document is NOT relevant (noise / wrong topic) → respond with exactly: no\n"
        "Do NOT add any explanation, punctuation, or extra words."
    ))
    human_msg = HumanMessage(content=(
        f"Retrieved document:\n\n{documents}\n\n"
        f"User question: {question}"
    ))
    # ──────────────────────────────────────────────────────────────────────

    response = llm.invoke([system_msg, human_msg])
    content  = response.content.strip().lower()
    grade    = "yes" if "yes" in content else "no"

    print(f"   Grade: {grade}")
    return {"needs_rewrite": grade}


# ---------------------------------------------------------------------------
# Task E – Final Generator
# ---------------------------------------------------------------------------
@retry_logic
def generate_node(state: AgentState) -> dict:
    print(colored("--- ✍️  GENERATE ---", "green"))
    question  = state["question"]
    documents = state["documents"]
    llm = get_llm()

    # ── Task E prompt ──────────────────────────────────────────────────────
    # Original rules (1-4): citation, honesty, precision, language.
    # Rule 5 added: dynamic time-period alignment — derive the target period
    # from the question itself and match it to the correct column in the table,
    # never mixing data across different time periods.
    system_content = (
        "You are a professional financial analyst assistant.\n\n"
        "Answer the user's question using ONLY the information in the provided context.\n\n"
        "Rules:\n"
        "1. Always cite your source using the format [Source: Apple 10-K] or "
        "[Source: Tesla 10-K] immediately after each fact.\n"
        "2. If the answer is not present in the context, respond with "
        "\"I don't know\" — never hallucinate or guess.\n"
        "3. Be precise: include exact figures and units (e.g., $391.04 billion).\n"
        "4. Answer in English.\n"
        "5. Financial tables often contain multiple time periods (e.g. columns for "
        "different years or quarters). First identify the specific time period the "
        "question is asking about, then use ONLY the values from that matching period. "
        "Never mix figures from different time periods in a single answer.\n\n"
        "Context:\n{context}"
    )
    # ──────────────────────────────────────────────────────────────────────

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_content),
        ("human", "{question}"),
    ])
    chain    = prompt | llm
    response = chain.invoke({"context": documents, "question": question})
    return {"generation": response.content}


# ---------------------------------------------------------------------------
# Task D – Query Rewriter
# ---------------------------------------------------------------------------
@retry_logic
def rewrite_node(state: AgentState) -> dict:
    print(colored("--- 🔄 REWRITE ---", "red"))
    question = state["question"]
    llm = get_llm()

    # ── Task D prompt ──────────────────────────────────────────────────────
    rewrite_prompt = HumanMessage(content=(
        "You are a financial query optimizer.\n"
        "The previous search query returned irrelevant results. "
        "Rewrite the query to use precise financial terminology so that a "
        "vector-database search over 10-K / earnings-report documents will "
        "find the correct passages.\n\n"
        "Transformation examples:\n"
        "  'how much did they spend on new tech'  →  'Research and Development expenses 2024'\n"
        "  'Apple money earned'                   →  'Apple total net sales fiscal year 2024'\n"
        "  'Tesla profit'                         →  'Tesla net income 2024 annual report'\n"
        "  'cost of making things'                →  'Cost of goods sold / Cost of sales 2024'\n\n"
        f"Original question: {question}\n\n"
        "Rewritten question (output ONLY the new question, nothing else):"
    ))
    # ──────────────────────────────────────────────────────────────────────

    response  = llm.invoke([rewrite_prompt])
    new_query = response.content.strip()
    print(f"   Rewritten: {new_query}")
    return {"question": new_query}


# ---------------------------------------------------------------------------
# LangGraph construction
# ---------------------------------------------------------------------------
def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve",        retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate",        generate_node)
    workflow.add_node("rewrite",         rewrite_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    def decide_to_generate(state: AgentState) -> str:
        if state["needs_rewrite"] == "yes":
            return "generate"
        if state["search_count"] > 2:
            print(colored("   Max retries reached – generating anyway.", "yellow"))
            return "generate"
        return "rewrite"

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    workflow.add_edge("rewrite",  "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()


def run_graph_agent(question: str) -> str:
    app    = build_graph()
    inputs = {
        "question":      question,
        "search_count":  0,
        "needs_rewrite": "no",
        "documents":     "",
        "generation":    "",
    }
    result = app.invoke(inputs)
    return result["generation"]


# ---------------------------------------------------------------------------
# Task A – Legacy ReAct agent
# ---------------------------------------------------------------------------
def run_legacy_agent(question: str) -> str:
    print(colored("--- 🤖 LEGACY AGENT (ReAct) ---", "magenta"))
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools.retriever import create_retriever_tool
    from langchain.tools.render import render_text_description

    tools = [
        create_retriever_tool(
            retriever,
            f"search_{key}_financials",
            f"Searches {key.capitalize()} Inc. financial documents (10-K / earnings report).",
        )
        for key, retriever in RETRIEVERS.items()
    ]

    if not tools:
        return "System Error: no retriever tools available."

    llm = get_llm()

    # ── Task A prompt ──────────────────────────────────────────────────────
    template = """You are a professional financial analyst assistant with access to Apple and Tesla financial documents.

You have access to the following tools:
{tools}

Use the following format STRICTLY:

Question: the input question you must answer
Thought: reason about what to do next
Action: the action to take, must be one of [{tool_names}]
Action Input: the specific query to send to the tool
Observation: the result returned by the tool
... (this Thought / Action / Action Input / Observation loop can repeat as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original question

MANDATORY RULES — violating any of these will cause test failures:
1. ENGLISH ONLY: Your Final Answer MUST be written in English, even if the question was asked in Chinese or another language.
2. YEAR PRECISION: Financial tables include columns for 2024, 2023, and 2022. You MUST use ONLY the 2024 column unless the question explicitly asks for a different year. Do not confuse values across years.
3. HONESTY: If the exact 2024 figure cannot be found in the retrieved documents, state "I don't know" clearly. Never guess, estimate, or substitute a value from a different year.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
    # ──────────────────────────────────────────────────────────────────────

    prompt = PromptTemplate.from_template(template)
    prompt = prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join(t.name for t in tools),
    )

    agent          = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )

    try:
        result = agent_executor.invoke({"input": question})
        return result["output"]
    except Exception as e:
        return f"Legacy Agent Error: {e}"
