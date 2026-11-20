"""
One place to create the OpenAI chat model and the OpenAI embedding model.

Why a dedicated file?
Creating model objects in ten different modules means ten places to update when
you change a model name, and ten chances to forget the API key check. Here we
create each model once, remember it, and hand the same object to everyone.
"""

from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI:
    """
    Create (or reuse) the OpenAI chat model used for every LLM call.

    The chat model does two jobs in this project: writing alternative versions
    of the user's question, and writing the final answer from the retrieved
    context.

    Returns:
        A configured LangChain ChatOpenAI model.

    Raises:
        ValueError: If OPENAI_API_KEY is missing.
    """
    config.check_openai_key()

    # @lru_cache means this function body runs only once per Python process.
    # Every later call returns the same object, so we never build a new HTTP
    # client for each question the user asks.
    return ChatOpenAI(
        model=config.OPENAI_CHAT_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=config.OPENAI_API_KEY,
    )


@lru_cache(maxsize=1)
def get_embedding_model() -> OpenAIEmbeddings:
    """
    Create (or reuse) the OpenAI embedding model used to build the vector index.

    An embedding turns a piece of text into a list of numbers (a vector).
    Texts with similar meaning get similar vectors, which is what makes
    "search by meaning" possible.

    Returns:
        A configured LangChain OpenAIEmbeddings model.

    Raises:
        ValueError: If OPENAI_API_KEY is missing.
    """
    config.check_openai_key()

    # Example:
    # embeddings.embed_query("refund policy") -> [0.013, -0.028, ...] (1536 numbers)
    return OpenAIEmbeddings(
        model=config.OPENAI_EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
    )
