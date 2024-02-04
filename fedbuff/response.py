from dataclasses import dataclass

@dataclass
class Response:
    reload: bool
    model_name: str