from transformers import pipeline
from pathlib import Path
from nicegui import ui, app
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import src.utils as utils

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RagBotNotInitialisedError(Exception):
    def __init__(self):
        self.message = "Run `RagBot.initialise()` method before querying."
        super().__init__(self.message)


class RagBot:
    def __init__(self, config_path):
        logger.info("Creating RagBot...")
        self._initialised = False
        logger.info(f"Loading config: {config_path}...")
        self._config = utils.load_config(config_path)
        self.VECTOR_DB_PATH = Path(self._config["vector_db_path"])
        self.EMBEDDING_MODEL_NAME = self._config["embedding_model_name"]
        self.DEVICE = self._config["device"]
        self.CHAT_MODEL_NAME = self._config["chat_model_name"]
        self.k = self._config["query"]["k"]

    def initialise(self):
        logger.info("Initialising Ragbot...")
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=self.EMBEDDING_MODEL_NAME,
            multi_process=False,
            model_kwargs={"device": self.DEVICE},
            # Set `True` for cosine similarity
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Loading vector DB...")
        self.KNOWLEDGE_VECTOR_DATABASE = FAISS.load_local(
            self.VECTOR_DB_PATH,
            self._embedding_model,
            allow_dangerous_deserialization=True,
        )
        logger.info("Loading LLM...")
        self._bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.CHAT_MODEL_NAME,
            device_map=self.DEVICE,
            quantization_config=self._bnb_config,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.CHAT_MODEL_NAME)
        self.LLM = pipeline(
            model=self._model,
            tokenizer=self._tokenizer,
            task=self._config["llm"]["task"],
            do_sample=self._config["llm"]["do_sample"],
            temperature=self._config["llm"]["temperature"],
            repetition_penalty=self._config["llm"]["repetition_penalty"],
            return_full_text=self._config["llm"]["return_full_text"],
            max_new_tokens=self._config["llm"]["max_new_tokens"],
        )
        self._prompt_in_chat_format = [
            {
                "role": "system",
                "content": """Using the information contained in the context, 
                give a comprehensive answer to the question.
                Respond only to the question asked, response should be concise and relevant to the question.
                Provide the number of the source document when relevant.
                If the answer cannot be deduced from the context, do not give an answer.""",
            },
            {
                "role": "user",
                "content": """Context: {context}
                Now here is the question you need to answer.
                Question: {question}""",
            },
        ]
        self.RAG_PROMPT_TEMPLATE = self._tokenizer.apply_chat_template(
            self._prompt_in_chat_format, tokenize=False, add_generation_prompt=True
        )
        self._initialised = True
        logger.info("RagBot successfully initialised...")

    def query(self, user_query: str):
        """Query the RagBot"""
        logger.info(f"Running query: {user_query}")
        # Check RagBot initialised
        if not self._initialised:
            err = RagBotNotInitialisedError
            logger.info(f"{err.__name__}: {err()}")
            raise err
        # Embed query and perform similarity search
        query_vector = self._embedding_model.embed_query(user_query)
        retrieved_docs = self.KNOWLEDGE_VECTOR_DATABASE.similarity_search(
            query=user_query, k=self.k
        )
        # Filter to the text only
        retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
        # Create the context for the prompt template
        context = "\nExtracted documents:\n"
        context += "".join(
            [
                f"Document {str(i)}:::\n" + doc
                for i, doc in enumerate(retrieved_docs_text)
            ]
        )
        # Format final prompt
        final_prompt = self.RAG_PROMPT_TEMPLATE.format(
            question="How to create a pipeline object?", context=context
        )
        # Generate answer
        answer = self.LLM(final_prompt)[0]["generated_text"]

        return answer


if __name__ == "__main__":
    # Create RagBot instance
    RagBot_instance = RagBot("config.yml")
    RagBot_instance.initialise()
    # Test
    RagBot_instance.query("How to build a pipeline?")
