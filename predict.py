import argparse
import logging
import os
from datetime import datetime
from gln.common.cmd_args import cmd_args as gln_args
from models.gln_model.gln_predictor import GLNPredictor
from models.localretro_model import localretro_parser
from models.localretro_model.localretro_predictor import LocalRetroPredictor
from models.neuralsym_model import neuralsym_parser
from models.neuralsym_model.neuralsym_predictor import NeuralSymPredictor
from models.retroxpert_model.retroxpert_predictor import RetroXpertPredictor
from models.transformer_model.transformer_predictor import TransformerPredictor
from onmt import opts as onmt_opts
from onmt.bin.translate import _get_parser as transformer_parser
from rdkit import RDLogger
from utils import misc


def get_predict_parser():
    parser = argparse.ArgumentParser("predict.py", conflict_handler="resolve")       # TODO: this is a hardcode
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


def predict_main(args, predict_parser):
    misc.log_args(args, message="Logging arguments")

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

        model_name = "gln"
        model_args = gln_args
        raw_data_files = [args.train_file, args.val_file, args.test_file]
        PredictorClass = GLNPredictor
    elif args.model_name == "localretro":
        localretro_parser.add_model_opts(predict_parser)
        localretro_parser.add_train_opts(predict_parser)
        localretro_parser.add_predict_opts(predict_parser)
        model_args, _unknown = predict_parser.parse_known_args()

        model_name = "localretro"
        PredictorClass = LocalRetroPredictor
    elif args.model_name == "transformer":
        # adapted from onmt.bin.translate.main()
        parser = transformer_parser()
        opt, _unknown = parser.parse_known_args()

        # update runtime args
        opt.config = args.config_file
        opt.log_file = args.log_file

        model_name = "transformer"
        model_args = opt
        PredictorClass = TransformerPredictor
    elif args.model_name == "retroxpert":
        onmt_opts.config_opts(predict_parser)
        onmt_opts.translate_opts(predict_parser)
        model_args, _unknown = predict_parser.parse_known_args()
        # update runtime args
        model_args.config = args.config_file
        model_args.log_file = args.log_file

        model_name = "retroxpert"
        PredictorClass = RetroXpertPredictor
    elif args.model_name == "neuralsym":
        neuralsym_parser.add_model_opts(predict_parser)
        neuralsym_parser.add_train_opts(predict_parser)
        neuralsym_parser.add_predict_opts(predict_parser)

        model_name = "neuralsym"
        model_args, _unknown = predict_parser.parse_known_args()
        raw_data_files = [args.test_file]
        PredictorClass = NeuralSymPredictor
    else:
        raise ValueError(f"Model {args.model_name} not supported!")

    logging.info("Start predicting")

    predictor = PredictorClass(
        model_name=model_name,
        model_args=model_args,
        model_config=model_config,
        data_name=data_name,
        raw_data_files=raw_data_files,
        processed_data_path=processed_data_path,
        model_path=model_path,
        test_output_path=test_output_path
    )
    predictor.predict()


if __name__ == "__main__":
    predict_parser = get_predict_parser()
    args, unknown = predict_parser.parse_known_args()

    # logger setup
    RDLogger.DisableLog("rdApp.warning")

    os.makedirs("./logs/predict", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    args.log_file = f"./logs/predict/{args.log_file}.{dt}"

    misc.setup_logger(args.log_file)

    # predict interface
    predict_main(args, predict_parser)
