import os
from abc import ABC, abstractmethod
from typing import Dict, List


class Trainer(ABC):
    """Base class for trainer"""

    @abstractmethod
    def __init__(self,
                 model_name: str,
                 config: Dict[str, any],                    # let's enforce everything to be passed in config
                 processed_data_files: List[str],
                 model_path: str):
        self.model_name = model_name
        self.config = config
        self.processed_data_files = processed_data_files,
        self.model_path = model_path

        for fn in processed_data_files:
            assert os.path.exists(fn), f"{fn} does not exist!"

        os.makedirs(model_path, exist_ok=True)
        self.model = self.build_train_model()

    @abstractmethod
    def build_train_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    # test() is optional
