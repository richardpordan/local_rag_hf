from nicegui import ui, app
from nicegui.events import KeyEventArguments
import datetime
import asyncio
import logging
import functools


# Logging setup
# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BotUI:
    def __init__(self, rag_bot):
        app.add_static_files("/static", "static")
        ui.add_head_html(
            """<link rel="stylesheet" type="text/css" href="/static/styles.css">"""
        )
        self._messages = [
            "Ask any question about this topic",
            "One sec, I'll think of one",
            "When you are ready, just send a message",
        ]
        self._whos = ["robot", "user", "robot"]
        self._thinking = False
        self._bot = rag_bot

    def _stamp_time_now(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _create_message(self, message: str, who: str = ["user", "robot"]):
        if who == "robot":
            name = "RAGBot"
            avatar = "https://robohash.org/ui"
            sent = False
        elif who == "user":
            name = "User"
            avatar = "https://robohash.org/human?set=set5"
            sent = True
        ui.chat_message(
            message,
            name=name,
            stamp=self._stamp_time_now(),
            avatar=avatar,
            sent=sent,
        )

    def _check_user_input(self, input_value):
        if input_value == None:
            check = True
        elif len(input_value) >= 5:
            check = True
        else:
            check = False

        return check

    def _append_new_msg_and_refresh(self, message: str, who: str = ["user", "robot"]):
        self._whos.append(who)
        self._messages.append(message)
        self.chat_ui.refresh()

    async def _get_answer_and_update_chat(self):
        logger.info("Getting answer and updating chat...")
        input_value = self._current_user_input.value
        if len(input_value) >= 5:
            self._current_user_input.set_value(None)
            self._current_user_input.disable()
            self._append_new_msg_and_refresh(message=input_value, who="user")
            self._append_new_msg_and_refresh(
                message="Just a minute, let me think...", who="robot"
            )
            # Run this bit async, otherwise will kill the GUI
            # Get the current event loop
            loop = asyncio.get_running_loop()
            # Run the blocking function in a thread pool and get a Future object
            future = loop.run_in_executor(
                None, functools.partial(self._bot.query, user_query=input_value)
            )
            # Wait for the Future object to complete and get the result
            response = await future
            self._append_new_msg_and_refresh(message=response, who="robot")
            self._current_user_input.enable()
        else:
            ui.notify(rf"'{input_value}' is not long enough.")

    @ui.refreshable
    def chat_ui(self):
        with ui.element("div").classes("container"):
            ui.label("RAGBot Chat").classes("title")
            with ui.element("div").classes("chat-box") as chat_box:
                for message_i, who_i in zip(self._messages, self._whos):
                    self._create_message(message=message_i, who=who_i)
            with ui.element("div").classes("input-area"):
                self._current_user_input = (
                    ui.input(
                        placeholder="Type your question here and hit enter",
                        validation={
                            "Too short": lambda value: self._check_user_input(value)
                        },
                    )
                    .props("rounded outlined dense")
                    .classes("input-box")
                    .on("keydown.enter", self._get_answer_and_update_chat)
                )

    def run_ui(self):
        ui.run()
