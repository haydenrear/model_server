import dataclasses

from aisuite.framework.message import Message


@dataclasses.dataclass(init=True)
class ChatModelArgs:
    model: str
    messages: list[Message]