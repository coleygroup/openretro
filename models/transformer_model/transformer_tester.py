import logging
import numpy as np
import os
import random
import torch
from onmt.bin.translate import translate as onmt_translate
from typing import Dict, List


class TransformerTester:
    """Class for Transformer Testing"""

    def __init__(self,
                 model_name: str,
                 model_args,
                 model_config: Dict[str, any],
                 data_name: str,
                 raw_data_files: List[str],
                 processed_data_path: str,
                 model_path: str,
                 test_output_path: str):

        self.model_name = model_name
        self.model_args = model_args
        self.model_config = model_config
        self.data_name = data_name
        self.processed_data_path = processed_data_path
        self.model_path = model_path
        self.test_output_path = test_output_path

        random.seed(self.model_args.seed)
        np.random.seed(self.model_args.seed)
        torch.manual_seed(self.model_args.seed)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info("Overwriting model args, (hardcoding essentially)")
        self.overwrite_model_args()
        logging.info(f"Updated model args: {self.model_args}")

    def overwrite_model_args(self):
        """Overwrite model args"""
        # Paths
        self.model_args.model = self.model_path
        self.model_args.src = os.path.join(self.processed_data_path, "src-text.txt")
        self.model_args.output = os.path.join(self.test_output_path, "predictions_on_test.txt")

    def test(self):
        """Actual file-based testing, a wrapper to onmt.bin.translate()"""
        onmt_translate(self.model_args)
