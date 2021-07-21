import argparse
import csv
import logging
import numpy as np
import os
import sys
from datetime import datetime
from rdkit import RDLogger
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

    # Load predictions and transform into a huge table {cano_prod: [cano_cand, ...]}
    predictions = {}
    with open(args.prediction_file, "r") as prediction_csv:
        prediction_reader = csv.DictReader(prediction_csv)
        for prediction_row in tqdm(prediction_reader):
            k = canonicalize_smiles(prediction_row["prod_smi"])
            v = []

            for i in range(n_best):
                try:
                    prediction = prediction_row[f"cand_precursor_{i+1}"]
                except KeyError:
                    break

                if not prediction or prediction == "9999":        # padding
                    break

                prediction = canonicalize_smiles(prediction)
                v.append(prediction)

            predictions[k] = v

    with open(args.test_file, "r") as test_csv:
        test_reader = csv.DictReader(test_csv)

        for i, test_row in enumerate(tqdm(test_reader)):
            gt, reagent, prod = test_row["rxn_smiles"].strip().split(">")
            k = canonicalize_smiles(prod)

            if k not in predictions:
                logging.info(f"Product {prod} not found in predictions (after canonicalization), skipping")
                continue

            gt = canonicalize_smiles(gt)
            for j, prediction in enumerate(predictions[k]):
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
    RDLogger.DisableLog("rdApp.*")

    os.makedirs("./logs/score", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    args.log_file = f"./logs/score/{args.log_file}.{dt}"

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