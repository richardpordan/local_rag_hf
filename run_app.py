import logging
from src.RagBot import RagBot
from src.RagBotFrontend import BotUI


if __name__ == "__main__":
    # Logging setup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # Create RagBot instance
    RagBot_instance = RagBot("config.yml")
    RagBot_instance.initialise()
    # Run UI
    bot_ui_instance = BotUI(rag_bot = RagBot_instance)
    bot_ui_instance.chat_ui()
    bot_ui_instance.run_ui()
