import argparse
import logging
import os
import sys
import time
from datetime import datetime
from gln.common.cmd_args import cmd_args as gln_args
from models.gln_model.gln_processor import GLNProcessor
from models.retroxpert_model import retroxpert_parser
from models.retroxpert_model.retroxpert_processor import RetroXpertProcessorS1, RetroXpertProcessorS2
from models.transformer_model.transformer_processor import TransformerProcessor
from onmt import opts as onmt_opts
from onmt.bin.preprocess import _get_parser as transformer_parser
from rdkit import RDLogger


def get_preprocess_parser():
    parser = argparse.ArgumentParser("preprocess.py", conflict_handler="resolve")       # TODO: this is a hardcode
    parser.add_argument("--model_name", help="model name", type=str, default="")
    parser.add_argument("--data_name", help="name of dataset, for easier reference", type=str, default="")
    parser.add_argument("--log_file", help="log file", type=str, default="")
    parser.add_argument("--config_file", help="model config file (optional)", type=str, default="")
    parser.add_argument("--train_file", help="train SMILES file", type=str, default="")
    parser.add_argument("--val_file", help="validation SMILES files", type=str, default="")
    parser.add_argument("--test_file", help="test SMILES files", type=str, default="")
    parser.add_argument("--processed_data_path", help="output path for processed data", type=str, default="")
    parser.add_argument("--num_cores", help="number of cpu cores to use", type=int, default=None)
    parser.add_argument("--stage", help="stage number (needed for RetroXpert)", type=int, default=0)

    return parser


def preprocess_main(args, preprocess_parser):
    start = time.time()

    model_name = ""
    model_args = None
    model_config = {}
    data_name = args.data_name
    raw_data_files = [args.train_file, args.val_file, args.test_file]
    processed_data_path = args.processed_data_path
    num_cores = args.num_cores

    if args.model_name == "gln":
        model_name = "gln"
        model_args = gln_args
        ProcessorClass = GLNProcessor
    elif args.model_name == "transformer":
        # adapted from onmt.bin.preprocess.main()
        parser = transformer_parser()
        opt, _unknown = parser.parse_known_args()

        # update runtime args
        opt.config = args.config_file
        opt.num_threads = args.num_cores
        opt.log_file = args.log_file

        model_name = "transformer"
        model_args = opt
        ProcessorClass = TransformerProcessor
    elif args.model_name == "retroxpert":
        retroxpert_parser.add_model_opts(preprocess_parser)
        retroxpert_parser.add_preprocess_opts(preprocess_parser)
        retroxpert_parser.add_train_opts(preprocess_parser)

        if args.stage == 1:
            model_name = "retroxpert_s1"
            ProcessorClass = RetroXpertProcessorS1
        elif args.stage == 2:
            model_name = "retroxpert_s2"
            raw_data_files = []
            onmt_opts.config_opts(preprocess_parser)
            onmt_opts.preprocess_opts(preprocess_parser)
            ProcessorClass = RetroXpertProcessorS2
        else:
            raise ValueError(f"--stage {args.stage} not supported! RetroXpert only has stages 1 and 2.")

        model_args, _unknown = preprocess_parser.parse_known_args()
    else:
        raise ValueError(f"Model {args.model_name} not supported!")

    processor = ProcessorClass(
        model_name=model_name,
        model_args=model_args,
        model_config=model_config,
        data_name=data_name,
        raw_data_files=raw_data_files,
        processed_data_path=processed_data_path,
        num_cores=num_cores
    )

    processor.check_data_format()
    processor.preprocess()
    logging.info(f"Preprocessing done, total time: {time.time() - start: .2f} s")
    sys.exit()              # from original gln, maybe to force python to exit correctly?


if __name__ == "__main__":
    preprocess_parser = get_preprocess_parser()
    args, unknown = preprocess_parser.parse_known_args()

    # logger setup
    RDLogger.DisableLog("rdApp.warning")

    os.makedirs("./logs/preprocess", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    args.log_file = f"./logs/preprocess/{args.log_file}.{dt}"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(args.log_file)
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    # preprocess interface
    preprocess_main(args, preprocess_parser)
