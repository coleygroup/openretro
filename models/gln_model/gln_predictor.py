import logging
import numpy as np
import os
import random
import torch
from gln.test.main_test import eval_model
from gln.test.model_inference import RetroGLN
from typing import Dict, List


class GLNPredictor:
    """Class for GLN Predicting"""

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
        self.train_file, self.val_file, self.test_file = raw_data_files
        self.processed_data_path = processed_data_path
        self.model_path = model_path
        self.test_output_path = test_output_path

        random.seed(model_args.seed)
        np.random.seed(model_args.seed)
        torch.manual_seed(model_args.seed)

        self.dropbox = processed_data_path
        self.best_model_idx = 0

    def predict(self):
        self.val()
        self.compile_into_csv()

    def val(self):
        """Core of test_all.sh and test_single.sh, adapted from test/main_test.py"""
        if self.model_args.test_all_ckpts:
            logging.info(f"test_all_ckpts flag set to True, testing all checkpoints")
            i = 0
            while True:
                model_dump = os.path.join(self.model_path, f"model-{i}.dump")
                if not os.path.isdir(model_dump):
                    logging.info(f"No checkpoints found at {model_dump}, exiting")
                    break

                logging.info(f"Checkpoints found at {model_dump}, building wrapper")
                self.model_args.model_for_test = model_dump
                model = RetroGLN(self.dropbox, self.model_args.model_for_test)

                logging.info(f"Testing {model_dump}")
                fname = self.val_file
                fname_pred = os.path.join(self.test_output_path, f"val-{i}.pred")
                eval_model("val", fname, model, fname_pred)
                i += 1
        else:
            model = RetroGLN(self.dropbox, self.model_args.model_for_test)

            logging.info(f"Testing {self.model_args.model_for_test}")
            for phase, fname in [("val", self.val_file),
                                 ("test", self.test_file)]:
                fname_pred = os.path.join(self.test_output_path, f"{phase}.pred")
                eval_model(phase, fname, model, fname_pred)

        # adapted from test/report_test_stats.py
        files = os.listdir(self.test_output_path)

        best_val = 0.0
        best_model_idx = 0
        for fname in files:
            if "val-" in fname and "summary" in fname:
                with open(os.path.join(self.test_output_path, fname), "r") as f:
                    f.readline()
                    top1 = float(f.readline().strip().split()[-1].strip())
                    if top1 > best_val:
                        best_val = top1
                        best_model_idx = fname.lstrip("val-").rstrip(".summary")

        logging.info(f"Best model idx: {best_model_idx}")
        model_dump = os.path.join(self.model_path, f"model-{best_model_idx}.dump")

        logging.info(f"Checkpoints found at {model_dump}, building wrapper")
        self.model_args.model_for_test = model_dump
        model = RetroGLN(self.dropbox, self.model_args.model_for_test)

        logging.info(f"Testing {model_dump}")
        fname = self.test_file
        fname_pred = os.path.join(self.test_output_path, f"test-{best_model_idx}.pred")
        eval_model("test", fname, model, fname_pred)

        self.best_model_idx = best_model_idx

    def compile_into_csv(self):
        logging.info("Compiling into predictions.csv")

        fname_pred = os.path.join(self.test_output_path, f"test-{self.best_model_idx}.pred")
        output_file = os.path.join(self.test_output_path, "predictions.csv")

        with open(fname_pred, "r") as f:
            line = f.readline()

        rxn_class, rxn_smi, n_best = line.strip().split()
        n_best = int(n_best)

        proposed_col_names = [f'cand_precursor_{i}' for i in range(1, n_best + 1)]
        headers = ['prod_smi']
        headers.extend(proposed_col_names)

        with open(fname_pred, "r") as f, open(output_file, "w") as of:
            header_line = ",".join(headers)
            of.write(f"{header_line}")

            for i, line in enumerate(f):
                items = line.strip().split()
                if len(items) == 3:         # meta line
                    rxn_class, rxn_smi, n_cand = items
                    reactants, reagent, product = rxn_smi.split(">")
                    of.write("\n")
                    of.write(product.strip())
                elif len(items) == 2:       # proposal line
                    template, reactants = items
                    of.write(",")
                    of.write(reactants.strip())
