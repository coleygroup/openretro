import os
from abc import ABC, abstractmethod
from typing import Dict, List


class Processor(ABC):
    """Base class for processor"""

    @abstractmethod
    def __init__(self,
                 model_name: str,
                 config: Dict[str, any],                    # let's enforce everything to be passed in config
                 raw_data_files: List[str],
                 processed_data_path: str):
        self.model_name = model_name
        self.config = config
        self.raw_data_files = raw_data_files
        self.processed_data_path = processed_data_path

    @abstractmethod
    def check_data_format(self) -> None:
        """Check that all files exists and the data format is correct for all"""
        for fn in self.raw_data_files:
            assert os.path.exists(fn), f"{fn} does not exist!"

    @abstractmethod
    def preprocess(self) -> None:
        """Actual file-based preprocessing"""
        pass
