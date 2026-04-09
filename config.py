import os
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from termcolor import colored

load_dotenv(override=True)

# ==============================================================================
# 1. Project Folders
# ==============================================================================
DATA_FOLDER = "data"
DB_FOLDER = "chroma_db"

# ==============================================================================
# 2. Dataset Configuration
# ==============================================================================
FILES = {
    "apple": "FY24_Q4_Consolidated_Financial_Statements.pdf",
    "tesla": "tsla-20241231-gen.pdf",
}

# ==============================================================================
# 3. Embedding Model Configuration (multi-model support for benchmarking)
# ==============================================================================
EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "mpnet":  "sentence-transformers/all-mpnet-base-v2",
    "gemma_embed": "google/embeddinggemma-300m"
}

# Active model is controlled by env var; defaults to "minilm"
# ACTIVE_EMBEDDING = os.getenv("EMBEDDING_MODEL", "gemma_embed")
# print(ACTIVE_EMBEDDING)
ACTIVE_EMBEDDING = "gemma_embed"

# ==============================================================================
# 4. Chunking Configuration (controllable via env vars for benchmarking)
# ==============================================================================
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "2000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "500"))
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 500


def get_embeddings(model_key: str | None = None) -> HuggingFaceEmbeddings:
    """Return a HuggingFace embedding model. Defaults to ACTIVE_EMBEDDING."""
    key = model_key or ACTIVE_EMBEDDING
    if key not in EMBEDDING_MODELS:
        raise ValueError(
            f"Unknown embedding model key '{key}'. "
            f"Available: {list(EMBEDDING_MODELS.keys())}"
        )
    model_name = EMBEDDING_MODELS[key]
    print(colored(f"🔄 Loading embedding model [{key}]: {model_name}", "cyan"))
    return HuggingFaceEmbeddings(model_name=model_name)


def get_db_folder(model_key: str | None = None) -> str:
    """Return the ChromaDB root path for a given embedding model key.

    Layout: chroma_db/<model_key>/<company>/
    This isolates indexes so multiple embedding experiments can coexist.
    """
    key = model_key or ACTIVE_EMBEDDING
    return os.path.join(DB_FOLDER, key)


# ==============================================================================
# 5. LLM Factory (supports google / openai / anthropic)
# ==============================================================================
def get_llm(temperature: float = 0):
    """Return a LangChain Chat model based on the LLM_PROVIDER env var."""
    provider = os.getenv("LLM_PROVIDER", "google").lower()

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print(colored("⚠️  GOOGLE_API_KEY not set!", "red"))
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
            temperature=temperature,
            google_api_key=api_key,
            convert_system_message_to_human=True,
            max_output_tokens=2048,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(colored("⚠️  OPENAI_API_KEY not set!", "red"))
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            api_key=api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print(colored("⚠️  ANTHROPIC_API_KEY not set!", "red"))
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
            temperature=temperature,
            api_key=api_key,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print(colored("⚠️  GROQ_API_KEY not set!", "red"))
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=temperature,
            api_key=api_key,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: '{provider}'")
