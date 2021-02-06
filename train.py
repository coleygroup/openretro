import argparse
import logging
import os
import sys
from datetime import datetime
from gln.common.cmd_args import cmd_args as gln_args
from models.gln_model.gln_trainer import GLNTrainer
from models.retroxpert_model import retroxpert_parser
from models.retroxpert_model.retroxpert_trainer import RetroXpertTrainerS1
from models.transformer_model.transformer_trainer import TransformerTrainer
from onmt.bin.train import _get_parser as transformer_parser
from rdkit import RDLogger


def get_train_parser():
    parser = argparse.ArgumentParser("train.py")
    parser.add_argument("--do_train", help="whether to do training (it's possible to only test)", action="store_true")
    parser.add_argument("--do_test", help="whether to do testing (only if implemented)", action="store_true")
    parser.add_argument("--model_name", help="model name", type=str, default="")
    parser.add_argument("--data_name", help="name of dataset, for easier reference", type=str, default="")
    parser.add_argument("--log_file", help="log file", type=str, default="")
    parser.add_argument("--config_file", help="model config file (optional)", type=str, default="")
    parser.add_argument("--train_file", help="train SMILES file", type=str, default="")
    parser.add_argument("--processed_data_path", help="output path for processed data", type=str, default="")
    parser.add_argument("--model_path", help="model output path", type=str, default="")

    return parser


def train_main(args):
    model_name = ""
    model_args = None
    model_config = {}
    data_name = args.data_name
    raw_data_files = []
    processed_data_path = args.processed_data_path
    model_path = args.model_path

    if args.model_name == "gln":
        model_name = "gln"
        model_args = gln_args
        raw_data_files = [args.train_file]
        TrainerClass = GLNTrainer
    elif args.model_name == "transformer":
        # adapted from onmt.bin.train.main()
        parser = transformer_parser()
        opt, _unknown = parser.parse_known_args()

        # update runtime args
        opt.config = args.config_file
        opt.log_file = args.log_file

        model_name = "transformer"
        model_args = opt
        TrainerClass = TransformerTrainer
    elif args.model_name == "retroxpert":
        retroxpert_parser.add_model_opts(train_parser)
        retroxpert_parser.add_train_opts(train_parser)
        model_args, _unknown = train_parser.parse_known_args()

        if args.stage == 1:
            model_name = "retroxpert_s1"
            TrainerClass = RetroXpertTrainerS1
        elif args.stage == 2:
            model_name = "retroxpert_s2"
            TrainerClass = RetroXpertTrainerS2
        else:
            raise ValueError(f"--stage {args.stage} not supported! RetroXpert only has stages 1 and 2.")
    else:
        raise ValueError(f"Model {args.model_name} not supported!")

    trainer = TrainerClass(
        model_name=model_name,
        model_args=model_args,
        model_config=model_config,
        data_name=data_name,
        raw_data_files=raw_data_files,
        processed_data_path=processed_data_path,
        model_path=model_path
    )

    logging.info("Building train model")
    trainer.build_train_model()

    if args.do_train:
        logging.info("Start training")
        trainer.train()
    if args.do_test:
        logging.info("Start testing")
        trainer.test()


if __name__ == "__main__":
    train_parser = get_train_parser()
    args, unknown = train_parser.parse_known_args()

    # logger setup
    RDLogger.DisableLog("rdApp.warning")

    os.makedirs("./logs/train", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    args.log_file = f"./logs/train/{args.log_file}.{dt}"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(args.log_file)
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    # train interface
    train_main(args, train_parser)
