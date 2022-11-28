import logging
import numpy as np
import os
import pandas as pd
import random
import torch
from dgllife.utils import WeaveAtomFeaturizer, CanonicalBondFeaturizer
from models.localretro_model.get_edit import write_edits
from models.localretro_model.LocalTemplate.template_decoder import read_prediction, decode_localtemplate
from models.localretro_model.model import LocalRetro
from models.localretro_model.utils import load_dataloader
from multiprocessing import Pool
from tqdm import tqdm
from typing import Dict, List
from utils import misc

global G_raw_predictions, G_atom_templates, G_bond_templates, G_template_infos


def get_k_predictions(tasks):
    """Adapted from Decode_predictions.get_k_predictions()"""
    global G_raw_predictions, G_atom_templates, G_bond_templates, G_template_infos

    test_id, top_k = tasks
    raw_predictions = G_raw_predictions[test_id]

    all_predictions = []
    product = raw_predictions[0]
    predictions = raw_predictions[1:]
    for prediction in predictions:
        mol, pred_site, template, template_info, score = \
            read_prediction(product, prediction, G_atom_templates, G_bond_templates, G_template_infos)
        local_template = '>>'.join(['(%s)' % smarts for smarts in template.split('_')[0].split('>>')])

        try:
            decoded_smiles = decode_localtemplate(mol, pred_site, local_template, template_info)
            if decoded_smiles is None or str((decoded_smiles, score)) in all_predictions:
                continue
        except Exception as e:
            # logging.info(e)
            continue
        all_predictions.append(decoded_smiles)

        if len(all_predictions) >= top_k:
            break

    return product, all_predictions


class LocalRetroPredictor:
    """Class for LocalRetro Predicting"""

    def __init__(self,
                 model_name: str,
                 model_args,
                 model_config: Dict[str, any],
                 data_name: str,
                 raw_data_files: List[str],
                 processed_data_path: str,
                 model_path: str,
                 test_output_path: str):

        self.model_name = model_name
        self.model_args = model_args
        self.model_config = model_config
        self.data_name = data_name
        self.test_file = raw_data_files[0]
        self.processed_data_path = processed_data_path
        self.model_path = model_path
        self.test_output_path = test_output_path

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.AtomTemplate_n = len(pd.read_csv(
            os.path.join(self.processed_data_path, "atom_templates.csv")))
        self.BondTemplate_n = len(pd.read_csv(
            os.path.join(self.processed_data_path, "bond_templates.csv")))

        logging.info("Overwriting model args, based on original localretro training script")
        self.overwrite_model_args()
        misc.log_args(self.model_args, message="Updated model args")

        random.seed(model_args.seed)
        np.random.seed(model_args.seed)
        torch.manual_seed(model_args.seed)

    def overwrite_model_args(self):
        self.model_args.mode = "test"
        self.model_args.device = self.device
        self.model_args.processed_data_path = self.processed_data_path
        self.model_args.model_path = self.model_path
        self.model_args.test_file = self.test_file
        self.model_args.test_output_path = self.test_output_path
        atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
                      'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
                      'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                      'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba',
                      'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs']
        self.model_args.node_featurizer = WeaveAtomFeaturizer(atom_types=atom_types)
        self.model_args.edge_featurizer = CanonicalBondFeaturizer(self_loop=True)

    def build_test_model(self):
        args = self.model_args
        self.model = LocalRetro(
            node_in_feats=args.node_featurizer.feat_size(),
            edge_in_feats=args.edge_featurizer.feat_size(),
            node_out_feats=args.node_out_feats,
            edge_hidden_feats=args.edge_hidden_feats,
            num_step_message_passing=args.num_step_message_passing,
            attention_heads=args.attention_heads,
            attention_layers=args.attention_layers,
            AtomTemplate_n=self.AtomTemplate_n,
            BondTemplate_n=self.BondTemplate_n
        )
        self.model = self.model.to(self.device)

        logging.info("Logging model summary")
        logging.info(self.model)
        logging.info(f"\nModel #Params: {sum([x.nelement() for x in self.model.parameters()]) / 1000} k")

        checkpoint = os.path.join(self.model_path, "LocalRetro.pth")
        state_dict = torch.load(checkpoint)
        self.model.load_state_dict(state_dict['model_state_dict'])
        logging.info(f"Loaded state_dict from {checkpoint}")

    def predict(self):
        self.get_raw_predictions()
        self.compile_into_csv()

    def get_raw_predictions(self):
        """Adapted from Test.py"""
        self.build_test_model()
        test_loader = load_dataloader(self.model_args)
        write_edits(self.model_args, self.model, test_loader)

    def compile_into_csv(self):
        """Adapted from Decode_predictions.py"""
        logging.info("Compiling into predictions.csv")
        global G_raw_predictions, G_atom_templates, G_bond_templates, G_template_infos

        args = self.model_args
        args.rxn_class_given = False

        G_atom_templates = pd.read_csv(os.path.join(self.processed_data_path, "atom_templates.csv"))
        G_bond_templates = pd.read_csv(os.path.join(self.processed_data_path, "bond_templates.csv"))
        G_template_infos = pd.read_csv(os.path.join(self.processed_data_path, "template_infos.csv"))

        G_atom_templates = {G_atom_templates['Class'][i]: G_atom_templates['Template'][i]
                            for i in G_atom_templates.index}
        G_bond_templates = {G_bond_templates['Class'][i]: G_bond_templates['Template'][i]
                            for i in G_bond_templates.index}
        G_template_infos = {G_template_infos['Template'][i]: {
            'edit_site': eval(G_template_infos['edit_site'][i]),
            'change_H': eval(G_template_infos['change_H'][i]),
            'change_C': eval(G_template_infos['change_C'][i]),
            'change_S': eval(G_template_infos['change_S'][i])}
            for i in G_template_infos.index}

        result_file = os.path.join(self.test_output_path, "raw_results.txt")
        output_file = os.path.join(self.test_output_path, "predictions.csv")

        G_raw_predictions = {}
        with open(result_file, 'r') as f:
            for line in f:
                seps = line.split('\t')
                if seps[0] == 'Test_id':
                    continue
                G_raw_predictions[int(seps[0])] = seps[1:]

        proposed_col_names = [f'cand_precursor_{i}' for i in range(1, args.top_k + 1)]
        headers = ['prod_smi']
        headers.extend(proposed_col_names)

        p = Pool()

        with open(output_file, "w") as of:
            header_line = ",".join(headers)
            of.write(f"{header_line}\n")

            tasks = [(i, args.top_k) for i in range(len(G_raw_predictions))]
            # DO NOT pass the args or results dict as in the original.
            # This will SIGNIFICANTLY slow down the code as everything needs to be serialized for p.imap
            for result in tqdm(p.imap(get_k_predictions, tasks),
                               total=len(G_raw_predictions),
                               desc="Decoding LocalRetro predictions and compiling into predictions.csv"):
                prod_smi, all_predictions = result
                of.write(prod_smi)
                of.write(",")
                of.write(",".join(all_predictions))
                of.write("\n")

        p.close()
        p.join()
