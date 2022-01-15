import glob
import logging
import numpy as np
import os
import random
import torch
from onmt.bin.translate import translate as onmt_translate
from typing import Dict, List
from utils import misc


class RetroXpertPredictor:
    """Class for RetroXpert Predicting"""

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
        misc.log_args(self.model_args, message="Updated model args")

    def overwrite_model_args(self):
        """Overwrite model args"""
        # Paths
        # Overwriting model path with the last checkpoint
        checkpoints = glob.glob(os.path.join(self.model_path, "model_step_*.pt"))
        last_checkpoint = sorted(checkpoints, reverse=True)[0]
        # self.model_args.models = [self.model_path]
        self.model_args.models = [last_checkpoint]
        self.model_args.src = os.path.join(self.processed_data_path, "opennmt_data_s2", "src-test-prediction.txt")
        self.model_args.output = os.path.join(self.test_output_path, "predictions_on_test.txt")

    def predict(self):
        """Actual file-based testing, a wrapper to onmt.bin.translate()"""
        if os.path.exists(self.model_args.output):
            logging.info(f"Results found at {self.model_args.output}, skip prediction.")
        else:
            onmt_translate(self.model_args)
        self.compile_into_csv()

    def compile_into_csv(self):
        logging.info("Compiling into predictions.csv")

        src_file = os.path.join(self.processed_data_path, "opennmt_data_s2", "src-test.txt")
        output_file = os.path.join(self.test_output_path, "predictions.csv")

        with open(src_file, "r") as f:
            total_src = sum(1 for _ in f)

        with open(self.model_args.output, "r") as f:
            total_gen = sum(1 for _ in f)

        n_best = self.model_args.n_best
        assert total_src == total_gen / n_best, \
            f"File length mismatch! Source total: {total_src}, " \
            f"prediction total: {total_gen}, n_best: {n_best}"

        proposed_col_names = [f'cand_precursor_{i}' for i in range(1, self.model_args.n_best + 1)]
        headers = ['prod_smi']
        headers.extend(proposed_col_names)

        with open(src_file, "r") as src_f, open(self.model_args.output, "r") as pred_f, open(output_file, "w") as of:
            header_line = ",".join(headers)
            of.write(f"{header_line}\n")

            for src_line in src_f:
                prod = src_line.split("[PREDICT]")[0]
                prods = prod.strip().split()[1:]        # drop the [RXN] token

                of.write("".join(prods))

                for j in range(n_best):
                    cand = pred_f.readline()
                    of.write(",")
                    of.write("".join(cand.strip().split()))
                of.write("\n")
