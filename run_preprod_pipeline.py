"""Pre-production processing to create the knowledgebase for RAG"""

import logging
from tqdm import tqdm
import pandas as pd
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import datasets
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter as RCTS
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
import src.utils as utils


def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RCTS.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in tqdm(knowledge_base):
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in tqdm(docs_processed):
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique


if __name__ == "__main__":
    # --- Logging setup ---
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.info("Running pre-production pipeline")
    # --- Config ---
    logger.info("Loading config")
    config = utils.load_config()
    VECTOR_DB_PATH = Path(config["vector_db_path"])
    EMBEDDING_MODEL_NAME = config["embedding_model_name"]
    DEVICE = config["device"]
    # We use a hierarchical list of separators specifically
    # tailored for splitting Markdown documents
    # This list is taken from LangChain's MarkdownTextSplitter
    MARKDOWN_SEPARATORS = [
        "\n#{1,6} ",
        "```\n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]
    # --- Data import and cleaning ---
    logger.info("Loading data")
    ds = datasets.load_dataset(config["hf_dataset"], split="train")
    RAW_KNOWLEDGE_BASE = [
        Document(page_content=doc["text"], metadata={"source": doc["source"]})
        for doc in tqdm(ds)
    ]
    docs_processed = split_documents(
        config["embedding_model_chunksize"],
        RAW_KNOWLEDGE_BASE,
        tokenizer_name=EMBEDDING_MODEL_NAME,
    )
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=False,
        model_kwargs={"device": DEVICE},
        encode_kwargs={
            "normalize_embeddings": True
        },  # Set `True` for cosine similarity
    )
    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        docs_processed,
        embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )
    KNOWLEDGE_VECTOR_DATABASE.save_local(VECTOR_DB_PATH)
