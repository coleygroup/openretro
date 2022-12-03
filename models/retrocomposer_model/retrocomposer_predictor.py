import json
import logging
import numpy as np
import os
import random
import torch
from models.retrocomposer_model.chemutils import cano_smiles
from models.retrocomposer_model.gnn import GNN_graphpred
from models.retrocomposer_model.prepare_mol_graph import MoleculeDataset
from torch_geometric.data import DataLoader
from tqdm import tqdm
from typing import Dict, List
from utils.chem_utils import canonicalize_smiles


def _eval_decoding(args, model, device, dataset, result_file, save_res=True):
    model.eval()
    pred_results = {}
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    # import pdb; pdb.set_trace()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            beam_nodes = model(batch, typed=False, decode=True, beam_size=args.beam_size)
            cano_pred_mols = {}
            for node in beam_nodes:
                batch_idx = node.index
                data_idx = batch.index[batch_idx]
                if data_idx not in cano_pred_mols:
                    cano_pred_mols[data_idx] = set()
                if data_idx not in pred_results:
                    pred_results[data_idx] = {
                        'rank': 1000,
                        'product': batch.product[batch_idx],
                        'reactant': batch.reactant[batch_idx],
                        'cano_reactants': batch.cano_reactants[batch_idx],
                        # 'type': batch.type[batch_idx].item(),
                        'seq_gt': batch.sequences[batch_idx],
                        'templates': batch.templates[batch_idx],
                        'templates_pred': [],
                        'templates_pred_log_prob': [],
                        'reactants_pred': [],
                        'seq_pred': [],
                    }
                # import ipdb; ipdb.set_trace()
                product = pred_results[data_idx]['product']
                seq_pred = node.targets_predict
                prod_smarts_list = []
                for i, cand in enumerate(batch.reaction_center_cands[batch_idx]):
                    if cand == seq_pred[0]:
                        prod_smarts_list.extend(batch.reaction_center_cands_smarts[batch_idx][i])
                prod_smarts_list = set(prod_smarts_list)
                assert len(prod_smarts_list)
                # keep product index unchanged, remove padding reactant indexes
                seq_pred[1:] = [tp for tp in seq_pred[1:] if tp < len(dataset.react_smarts_list)]
                decoded_results = dataset.decode_reactant_from_seq(
                    product, seq_pred, prod_smarts_list, keep_mapnums=True)
                for decoded_result in decoded_results:
                    pred_tmpl, pred_mols = decoded_result
                    for pred_mol in pred_mols:
                        cano_pred_mol = cano_smiles(pred_mol)
                        if cano_pred_mol not in cano_pred_mols[data_idx]:
                            cano_pred_mols[data_idx].add(cano_pred_mol)
                            pred_results[data_idx]['templates_pred_log_prob'].append(node.log_prob.item())
                            pred_results[data_idx]['templates_pred'].append(pred_tmpl)
                            pred_results[data_idx]['reactants_pred'].append(pred_mol)
                            pred_results[data_idx]['seq_pred'].append(seq_pred)
                            if pred_results[data_idx]['cano_reactants'] == cano_pred_mol:
                                pred_results[data_idx]['rank'] = min(
                                    pred_results[data_idx]['rank'], len(pred_results[data_idx]['seq_pred']))

            beam_nodes.clear()

    logging.info(f'Total examples to evaluate: {len(dataset)}')
    ranks = [val['rank'] == 1 for val in pred_results.values()]
    logging.info(f'approximate top1 lower bound: {np.mean(ranks)}, {np.sum(ranks)}/{len(ranks)}')
    if save_res:
        with open(result_file, 'w') as f:
            json.dump(pred_results, f, indent=4)


class RetroComposerPredictor:
    """Class for RetroComposer Predicting"""

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
        self.processed_data_path = processed_data_path
        self.model_path = model_path
        self.test_output_path = test_output_path

        random.seed(self.model_args.seed)
        np.random.seed(self.model_args.seed)
        torch.manual_seed(self.model_args.seed)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_dataset = MoleculeDataset(
            root=processed_data_path, split='test', load_mol=True)
        self.prod_word_size = len(self.test_dataset.prod_smarts_fp_list)
        self.react_word_size = len(self.test_dataset.react_smarts_list)

    def build_test_model(self):
        args = self.model_args
        self.model = GNN_graphpred(
            args.num_layer,
            args.emb_dim,
            args.atom_feat_dim,
            args.bond_feat_dim,
            args.center_loss_type,
            0,
            self.prod_word_size,
            self.react_word_size,
            JK=args.JK,
            drop_ratio=args.dropout_ratio,
            graph_pooling=args.graph_pooling
        )
        del self.model.gnn_diff
        del self.model.scoring
        self.model = self.model.to(self.device)

        logging.info("Logging model summary")
        logging.info(self.model)
        logging.info(f"\nModel #Params: {sum([x.nelement() for x in self.model.parameters()]) / 1000} k")

    def predict(self):
        self.eval_decoding()
        self.compile_into_csv()

    def eval_decoding(self):
        """Adapted from run_retro.py"""
        result_file = os.path.join(self.test_output_path, 'beam_result.json')
        if os.path.exists(result_file):
            logging.info(f"Results found at {result_file}, skip prediction.")
            return

        self.build_test_model()
        model_file = os.path.join(self.model_path, "model.pt")
        self.model.from_pretrained(model_file)
        logging.info(f"Loaded model from {model_file}")
        self.model.eval()

        _eval_decoding(self.model_args, self.model, self.device, self.test_dataset, result_file=result_file)

    def compile_into_csv(self):
        logging.info("Compiling into predictions.csv")
        n_best = 50

        result_file = os.path.join(self.test_output_path, 'beam_result.json')
        output_file = os.path.join(self.test_output_path, "predictions.csv")

        with open(result_file, 'r') as f:
            results = json.load(f)

        proposed_col_names = [f'cand_precursor_{i}' for i in range(1, n_best + 1)]
        headers = ['prod_smi']
        headers.extend(proposed_col_names)

        with open(output_file, "w") as of:
            header_line = ",".join(headers)
            of.write(f"{header_line}\n")

            for idx, result in tqdm(results.items()):
                prod = canonicalize_smiles(result["product"], remove_atom_number=True)
                of.write(prod)
                for j, cand in enumerate(result["reactants_pred"]):
                    if j > n_best:
                        break
                    of.write(",")
                    of.write(canonicalize_smiles(cand))
                of.write("\n")
