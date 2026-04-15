from typing import Optional


class BaseDataLoader:
    def __init__(self, opt) -> None:
        self.opt = opt

    def load_data(self) -> Optional[object]:
        return None
