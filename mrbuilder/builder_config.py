from typing import Callable


class BuilderConfig:
    def __init__(self) -> None:
        super().__init__()

    def get_model_creator(self) -> Callable:
        pass

    def get_model_input_builder(self) -> Callable:
        pass
