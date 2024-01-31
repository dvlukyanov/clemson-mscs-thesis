from dataclasses import dataclass

@dataclass
class Message:
    sender_id: int
    model_name: str
    iteration: int