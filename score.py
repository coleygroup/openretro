import argparse
import csv
import logging
import numpy as np
import os
import sys
from datetime import datetime
from tqdm import tqdm
from utils.chem_utils import canonicalize_smiles


def get_score_parser():
    parser = argparse.ArgumentParser("score.py", conflict_handler="resolve")
    parser.add_argument("--model_name", help="model name", type=str, default="")
    parser.add_argument("--log_file", help="log file", type=str, default="")
    parser.add_argument("--test_file", help="test SMILES file", type=str, default="")
    parser.add_argument("--prediction_file", help="prediction file", type=str, default="")

    return parser


def score_main(args):
    """Adapted from Molecular Transformer"""
    n_best = 50

    logging.info(f"Scoring predictions with model: {args.model_name}")
    with open(args.test_file, "r") as test_csv:
        total = sum(1 for _ in test_csv) - 1

    accuracies = np.zeros([total, n_best], dtype=np.float32)

    with open(args.test_file, "r") as test_csv, open(args.prediction_file, "r") as prediction_csv:
        test_reader = csv.DictReader(test_csv)
        prediction_reader = csv.DictReader(prediction_csv)

        for i, (test_row, prediction_row) in enumerate(tqdm(zip(test_reader, prediction_reader))):
            gt, reagent, prod = test_row["reactants>reagents>production"].strip().split(">")

            assert canonicalize_smiles(prod) == canonicalize_smiles(prediction_row["prod_smi"]), \
                f"Product mismatch on row {i}! Given: {prod}, for generation: {prediction_row['prod_smi']}"

            gt = canonicalize_smiles(gt)
            for j in range(n_best):
                prediction = prediction_row[f"cand_precursor_{j+1}"]
                if prediction == "9999":        # padding
                    break

                prediction = canonicalize_smiles(prediction)

                if prediction == gt:
                    accuracies[i, j:] = 1.0
                    break

    # Log statistics
    mean_accuracies = np.mean(accuracies, axis=0)
    for n in range(n_best):
        logging.info(f"Top {n+1} accuracy: {mean_accuracies[n]}")


if __name__ == "__main__":
    score_parser = get_score_parser()
    args, unknown = score_parser.parse_known_args()

    # logger setup
    os.makedirs("./logs/score", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    args.log_file = f"./logs/predict/{args.log_file}.{dt}"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(args.log_file)
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    # score interface
    score_main(args)
