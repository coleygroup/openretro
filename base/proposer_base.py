import os
from abc import ABC, abstractmethod
from typing import Dict, List


class Proposer(ABC):
    """Base class for proposer"""

    @abstractmethod
    def __init__(self,
                 model_name: str,
                 config: Dict[str, any],                    # let's enforce everything to be passed in config
                 model_path: str):
        self.model_name = model_name
        self.config = config
        self.model_path = model_path

        assert os.path.exists(model_path), f"{model_path} does not exist!"

        self.model = self.build_predict_model()

    @abstractmethod
    def build_predict_model(self):
        pass

    @abstractmethod
    def propose(self, input_smiles: List[str], **kwargs):
        pass

    # propose_batch() is optional
