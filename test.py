import argparse
import logging
import os
import sys
from datetime import datetime
from gln.common.cmd_args import cmd_args as gln_args
from models.gln_model.gln_tester import GLNTester
from rdkit import RDLogger


def parse_args():
    parser = argparse.ArgumentParser("test.py")
    parser.add_argument("--test_all_ckpts", help="whether to test all checkpoints", action="store_true")
    parser.add_argument("--model_name", help="model name", type=str, default="")
    parser.add_argument("--data_name", help="name of dataset, for easier reference", type=str, default="")
    parser.add_argument("--log_file", help="log file", type=str, default="")
    parser.add_argument("--train_file", help="train SMILES file", type=str, default="")
    parser.add_argument("--val_file", help="validation SMILES files", type=str, default="")
    parser.add_argument("--test_file", help="test SMILES files", type=str, default="")
    parser.add_argument("--processed_data_path", help="output path for processed data", type=str, default="")
    parser.add_argument("--model_path", help="model output path", type=str, default="")
    parser.add_argument("--test_output_path", help="test output path", type=str, default="")

    return parser.parse_args()


def test_main(args):
    """Simplified interface for testing only. For actual usage downstream use the respective proposer class"""
    os.makedirs(args.test_output_path, exist_ok=True)

    if args.model_name == "gln":
        # Overwrite default gln_args with runtime args
        gln_args.test_all_ckpts = args.test_all_ckpts
        tester = GLNTester(model_name="gln",
                           model_args=gln_args,
                           model_config={},
                           data_name=args.data_name,
                           raw_data_files=[args.train_file, args.val_file, args.test_file],
                           processed_data_path=args.processed_data_path,
                           model_path=args.model_path,
                           test_output_path=args.test_output_path)

    else:
        raise ValueError(f"Model {args.model_name} not supported!")

    logging.info("Start testing")
    tester.test()


if __name__ == "__main__":
    args = parse_args()

    # logger setup
    RDLogger.DisableLog("rdApp.warning")

    os.makedirs("./logs/test", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(f"./logs/test/{args.log_file}.{dt}")
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    # preprocess interface
    test_main(args)
