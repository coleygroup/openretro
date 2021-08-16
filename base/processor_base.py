import csv
import logging
import os
from abc import ABC, abstractmethod
from rdkit import Chem
from typing import Dict, List


class Processor(ABC):
    """Base class for processor"""

    @abstractmethod
    def __init__(self,
                 model_name: str,
                 model_args,                                        # let's enforce everything to be passed in args
                 model_config: Dict[str, any],                      # or config
                 data_name: str,
                 raw_data_files: List[str],
                 processed_data_path: str):
        self.model_name = model_name
        self.model_args = model_args
        self.model_config = model_config
        self.data_name = data_name
        self.raw_data_files = raw_data_files
        self.processed_data_path = processed_data_path

        os.makedirs(self.processed_data_path, exist_ok=True)

        self.check_count = 100

    def check_data_format(self) -> None:
        """Check that all files exists and the data format is correct for the first few lines"""
        logging.info(f"Checking the first {self.check_count} entries for each file")
        for fn in self.raw_data_files:
            if not fn:
                continue
            assert os.path.exists(fn), f"{fn} does not exist!"

            with open(fn, "r") as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for i, row in enumerate(csv_reader):
                    if i > self.check_count:  # check the first few rows
                        break

                    assert (c in row for c in ["class", "rxn_smiles"]), \
                        f"Error processing file {fn} line {i}, ensure columns 'class' and " \
                        f"'rxn_smiles' is included!"

                    reactants, reagents, products = row["rxn_smiles"].split(">")
                    Chem.MolFromSmiles(reactants)  # simply ensures that SMILES can be parsed
                    Chem.MolFromSmiles(products)  # simply ensures that SMILES can be parsed

        logging.info("Data format check passed")

    @abstractmethod
    def preprocess(self) -> None:
        """Actual file-based preprocessing"""
        pass
