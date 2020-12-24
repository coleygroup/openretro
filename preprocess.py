import argparse
import logging
import os
import sys
import time
from datetime import datetime
from gln.common.cmd_args import cmd_args as gln_args
from models.gln_model.gln_processor import GLNProcessor
from rdkit import RDLogger


def parse_args():
    parser = argparse.ArgumentParser("preprocess.py")
    parser.add_argument("--model_name", help="model name", type=str, default="")
    parser.add_argument("--data_name", help="name of dataset, for easier reference", type=str, default="")
    parser.add_argument("--log_file", help="log file", type=str, default="")
    parser.add_argument("--train_file", help="train SMILES file", type=str, default="")
    parser.add_argument("--val_file", help="validation SMILES files", type=str, default="")
    parser.add_argument("--test_file", help="test SMILES files", type=str, default="")
    parser.add_argument("--processed_data_path", help="output path for processed data", type=str, default="")
    parser.add_argument("--num_cores", help="number of cpu cores to use", type=int, default=None)

    return parser.parse_args()


def preprocess_main(args):
    start = time.time()

    if args.model_name == "gln":
        processor = GLNProcessor(model_name="gln",
                                 model_args=gln_args,
                                 model_config={},
                                 data_name=args.data_name,
                                 raw_data_files=[args.train_file, args.val_file, args.test_file],
                                 processed_data_path=args.processed_data_path,
                                 num_cores=args.num_cores)

    else:
        raise ValueError(f"Model {args.model_name} not supported!")

    processor.check_data_format()
    processor.preprocess()
    logging.info(f"Preprocessing done, total time: {time.time() - start: .2f} s")
    sys.exit()              # from original, maybe to force python to exit correctly?


if __name__ == "__main__":
    args = parse_args()

    # logger setup
    RDLogger.DisableLog("rdApp.warning")

    os.makedirs("./logs/preprocess", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(f"./logs/preprocess/{args.log_file}.{dt}")
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    # preprocess interface
    preprocess_main(args)