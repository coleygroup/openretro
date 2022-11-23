import argparse
import logging
import os
import sys
from datetime import datetime

from gln.common.cmd_args import cmd_args as gln_args
from models.gln_model.gln_trainer import GLNTrainer
from models.localretro_model import localretro_parser
from models.localretro_model.localretro_trainer import LocalRetroTrainer
from models.neuralsym_model import neuralsym_parser
from models.neuralsym_model.neuralsym_trainer import NeuralSymTrainer
from models.retroxpert_model import retroxpert_parser
from models.retroxpert_model.retroxpert_trainer import RetroXpertTrainerS1
from models.transformer_model.transformer_trainer import TransformerTrainer
from onmt import opts as onmt_opts
from onmt.bin.train import _get_parser as transformer_parser
from rdkit import RDLogger
from utils import misc


def get_train_parser():
    parser = argparse.ArgumentParser("train.py", conflict_handler="resolve")       # TODO: this is a hardcode
    parser.add_argument("--do_train", help="whether to do training (it's possible to only test)", action="store_true")
    parser.add_argument("--model_name", help="model name", type=str, default="")
    parser.add_argument("--data_name", help="name of dataset, for easier reference", type=str, default="")
    parser.add_argument("--log_file", help="log file", type=str, default="")
    parser.add_argument("--config_file", help="model config file (optional)", type=str, default="")
    parser.add_argument("--train_file", help="train SMILES file", type=str, default="")
    parser.add_argument("--processed_data_path", help="output path for processed data", type=str, default="")
    parser.add_argument("--model_path", help="model output path", type=str, default="")
    parser.add_argument("--stage", help="stage number (needed for RetroXpert)", type=int, default=0)

    return parser


def train_main(args, train_parser):
    misc.log_args(args, message="Logging arguments")

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
    elif args.model_name == "localretro":
        localretro_parser.add_model_opts(train_parser)
        localretro_parser.add_train_opts(train_parser)

        model_name = "localretro"
        model_args, _unknown = train_parser.parse_known_args()
        TrainerClass = LocalRetroTrainer
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

        if args.stage == 1:
            model_name = "retroxpert_s1"
            retroxpert_parser.add_train_opts(train_parser)
            TrainerClass = RetroXpertTrainerS1
        elif args.stage == 2:
            model_name = "retroxpert_s2"
            onmt_opts.config_opts(train_parser)
            onmt_opts.model_opts(train_parser)
            onmt_opts.train_opts(train_parser)
            TrainerClass = TransformerTrainer
        else:
            raise ValueError(f"--stage {args.stage} not supported! RetroXpert only has stages 1 and 2.")

        model_args, _unknown = train_parser.parse_known_args()
        # update runtime args
        model_args.config = args.config_file
        model_args.log_file = args.log_file
    elif args.model_name == "neuralsym":
        neuralsym_parser.add_model_opts(train_parser)
        neuralsym_parser.add_train_opts(train_parser)

        model_name = "neuralsym"
        model_args, _unknown = train_parser.parse_known_args()
        TrainerClass = NeuralSymTrainer
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


if __name__ == "__main__":
    train_parser = get_train_parser()
    args, unknown = train_parser.parse_known_args()

    # logger setup
    RDLogger.DisableLog("rdApp.warning")

    os.makedirs("./logs/train", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    args.log_file = f"./logs/train/{args.log_file}.{dt}"

    misc.setup_logger(args.log_file)

    # train interface
    train_main(args, train_parser)
