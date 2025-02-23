"""Run the RAG Chat Bot app file"""

from src.RagBotLLMBackend import RagBot
from src.RagBotFrontend import BotUI
from src import utils


# Logging setup
logger = utils.create_logger()


if __name__ in {"__main__", "__mp_main__"}:
    # RAG LLM backend setup
    logger.info("Creating RagBot *instance*...")
    RagBot_instance = RagBot("config.yml")
    RagBot_instance.initialise()
    # App GUI
    logger.info("Initialising GUI")
    bot_ui_instance = BotUI(rag_bot=RagBot_instance)
    bot_ui_instance.chat_ui()
    bot_ui_instance.run_ui()
