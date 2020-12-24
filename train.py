import argparse
import logging
import os
import sys
from datetime import datetime
from gln.common.cmd_args import cmd_args as gln_args
from models.gln_model.gln_trainer import GLNTrainer
from rdkit import RDLogger


def parse_args():
    parser = argparse.ArgumentParser("train.py")
    parser.add_argument("--do_train", help="whether to do training (it's possible to only test)", action="store_true")
    parser.add_argument("--do_test", help="whether to do testing (only if implemented)", action="store_true")
    parser.add_argument("--model_name", help="model name", type=str, default="")
    parser.add_argument("--data_name", help="name of dataset, for easier reference", type=str, default="")
    parser.add_argument("--log_file", help="log file", type=str, default="")
    parser.add_argument("--train_file", help="train SMILES file", type=str, default="")
    parser.add_argument("--processed_data_path", help="output path for processed data", type=str, default="")
    parser.add_argument("--model_path", help="model output path", type=str, default="")

    return parser.parse_args()


def train_main(args):
    if args.model_name == "gln":
        trainer = GLNTrainer(model_name="gln",
                             model_args=gln_args,
                             model_config={},
                             data_name=args.data_name,
                             raw_data_files=[args.train_file],
                             processed_data_path=args.processed_data_path,
                             model_path=args.model_path)

    else:
        raise ValueError(f"Model {args.model_name} not supported!")

    logging.info("Building train model")
    trainer.build_train_model()

    if args.do_train:
        logging.info("Start training")
        trainer.train()
    if args.do_test:
        logging.info("Start testing")
        trainer.test()


if __name__ == "__main__":
    args = parse_args()

    # logger setup
    RDLogger.DisableLog("rdApp.warning")

    os.makedirs("./logs/train", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(f"./logs/train/{args.log_file}.{dt}")
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    # preprocess interface
    train_main(args)