import argparse
import logging
import os
import sys
from datetime import datetime
from gln.common.cmd_args import cmd_args as gln_args
from models.gln_model.gln_tester import GLNTester
from models.retroxpert_model import retroxpert_parser
from models.retroxpert_model.retroxpert_tester import RetroXpertTester
from models.transformer_model.transformer_tester import TransformerTester
from onmt.bin.translate import _get_parser as transformer_parser
from rdkit import RDLogger


def get_test_parser():
    parser = argparse.ArgumentParser("test.py")
    parser.add_argument("--test_all_ckpts", help="whether to test all checkpoints", action="store_true")
    parser.add_argument("--model_name", help="model name", type=str, default="")
    parser.add_argument("--data_name", help="name of dataset, for easier reference", type=str, default="")
    parser.add_argument("--log_file", help="log file", type=str, default="")
    parser.add_argument("--config_file", help="model config file (optional)", type=str, default="")
    parser.add_argument("--train_file", help="train SMILES file", type=str, default="")
    parser.add_argument("--val_file", help="validation SMILES files", type=str, default="")
    parser.add_argument("--test_file", help="test SMILES files", type=str, default="")
    parser.add_argument("--processed_data_path", help="output path for processed data", type=str, default="")
    parser.add_argument("--model_path", help="model output path", type=str, default="")
    parser.add_argument("--test_output_path", help="test output path", type=str, default="")

    return parser


def test_main(args, test_parser):
    """Simplified interface for testing only. For actual usage downstream use the respective proposer class"""
    os.makedirs(args.test_output_path, exist_ok=True)

    model_name = ""
    model_args = None
    model_config = {}
    data_name = args.data_name
    raw_data_files = []
    processed_data_path = args.processed_data_path
    model_path = args.model_path
    test_output_path = args.test_output_path

    if args.model_name == "gln":
        # Overwrite default gln_args with runtime args
        gln_args.test_all_ckpts = args.test_all_ckpts

        model_name = "gln",
        model_args = gln_args,
        raw_data_files = [args.train_file, args.val_file, args.test_file]
        TesterClass = GLNTester
    elif args.model_name == "transformer":
        # adapted from onmt.bin.translate.main()
        parser = transformer_parser()
        opt, _unknown = parser.parse_known_args()

        # update runtime args
        opt.config = args.config_file
        opt.log_file = args.log_file

        model_name = "transformer"
        model_args = opt
        TesterClass = TransformerTester
    elif args.model_name == "retroxpert":
        # retroxpert_parser.add_model_opts(test_parser)
        # retroxpert_parser.add_train_opts(test_parser)
        model_args, _unknown = test_parser.parse_known_args()

        model_name = "retroxpert"
        TesterClass = RetroXpertTester

    else:
        raise ValueError(f"Model {args.model_name} not supported!")

    logging.info("Start testing")

    tester = TesterClass(
        model_name=model_name,
        model_args=model_args,
        model_config=model_config,
        data_name=data_name,
        raw_data_files=raw_data_files,
        processed_data_path=processed_data_path,
        model_path=model_path,
        test_output_path=test_output_path
    )
    tester.test()


if __name__ == "__main__":
    test_parser = get_test_parser()
    args, unknown = test_parser.parse_known_args()

    # logger setup
    RDLogger.DisableLog("rdApp.warning")

    os.makedirs("./logs/test", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    args.log_file = f"./logs/test/{args.log_file}.{dt}"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(args.log_file)
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    # test interface
    test_main(args, test_parser)
