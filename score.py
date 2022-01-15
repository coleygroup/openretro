import argparse
import csv
import logging
import multiprocessing
import numpy as np
import os
import sys
from datetime import datetime
from rdkit import RDLogger
from tqdm import tqdm
from utils.chem_utils import canonicalize_smiles
from utils import misc

global G_predictions


def get_score_parser():
    parser = argparse.ArgumentParser("score.py", conflict_handler="resolve")
    parser.add_argument("--model_name", help="model name", type=str, default="")
    parser.add_argument("--log_file", help="log file", type=str, default="")
    parser.add_argument("--test_file", help="test SMILES file", type=str, default="")
    parser.add_argument("--prediction_file", help="prediction file", type=str, default="")
    parser.add_argument("--num_cores", help="number of cpu cores to use", type=int, default=4)

    return parser


def csv2kv(_args):
    prediction_row, n_best = _args
    k = canonicalize_smiles(prediction_row["prod_smi"])
    v = []

    for i in range(n_best):
        try:
            prediction = prediction_row[f"cand_precursor_{i + 1}"]
        except KeyError:
            break

        if not prediction or prediction == "9999":          # padding
            break

        prediction = canonicalize_smiles(prediction)
        v.append(prediction)

    return k, v


def match_results(_args):
    global G_predictions
    test_row, n_best = _args
    predictions = G_predictions

    accuracy = np.zeros(n_best, dtype=np.float32)

    gt, reagent, prod = test_row["rxn_smiles"].strip().split(">")
    k = canonicalize_smiles(prod)

    if k not in predictions:
        logging.info(f"Product {prod} not found in predictions (after canonicalization), skipping")
        return accuracy

    gt = canonicalize_smiles(gt)
    for j, prediction in enumerate(predictions[k]):
        if prediction == gt:
            accuracy[j:] = 1.0
            break

    return accuracy


def score_main(args):
    """
        Adapted from Molecular Transformer
        Parallelized (210826 by ztu)
    """
    global G_predictions
    n_best = 50

    logging.info(f"Scoring predictions with model: {args.model_name}")

    # Load predictions and transform into a huge table {cano_prod: [cano_cand, ...]}
    logging.info(f"Loading predictions from {args.prediction_file}")
    predictions = {}
    p = multiprocessing.Pool(args.num_cores)

    with open(args.prediction_file, "r") as prediction_csv:
        prediction_reader = csv.DictReader(prediction_csv)
        for result in tqdm(p.imap(csv2kv,
                                  ((prediction_row, n_best) for prediction_row in prediction_reader))):
            k, v = result
            predictions[k] = v

    G_predictions = predictions

    p.close()
    p.join()
    p = multiprocessing.Pool(args.num_cores)        # re-initialize to see the global variable

    # Results matching
    logging.info(f"Matching against ground truth from {args.test_file}")
    with open(args.test_file, "r") as test_csv:
        test_reader = csv.DictReader(test_csv)
        accuracies = p.imap(match_results,
                            ((test_row, n_best) for test_row in test_reader))
        accuracies = np.stack(list(accuracies))

    p.close()
    p.join()

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

    misc.setup_logger(args.log_file)

    # score interface
    score_main(args)
