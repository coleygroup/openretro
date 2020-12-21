import csv
from base.processor_base import Processor
from rdkit import Chem
from typing import Dict, List


class GLNProcessor(Processor):
    """Class for GLN Preprocessing"""

    def __init__(self,
                 model_name: str,
                 config: Dict[str, any],
                 raw_data_files: List[str],
                 processed_data_path: str):
        super().__init__(model_name=model_name,
                         config=config,
                         raw_data_files=raw_data_files,
                         processed_data_path=processed_data_path)
        self.check_count = 100

    def check_data_format(self) -> None:
        """Check that all files exists and the data format is correct for all"""
        super().check_data_format()
        for fn in self.raw_data_files:
            pass


    def preprocess(self) -> None:
        """Actual file-based preprocessing"""
