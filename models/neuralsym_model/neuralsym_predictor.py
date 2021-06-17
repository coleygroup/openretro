import csv
import logging
import multiprocessing
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn as nn
from collections import Counter
from functools import partial
from models.neuralsym_model.dataset import FingerprintDataset
from models.neuralsym_model.model import TemplateNN_Highway, TemplateNN_FC
from rdchiral.main import rdchiralReaction, rdchiralReactants, rdchiralRun
from rdkit import Chem
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List


def gen_precs(templates_filtered, preds, phase_topk, task):
    i, prod_smi_nomap = task
    # generate predictions from templates
    precursors, dup_count = [], 0
    pred_temp_idxs = preds[i]
    for idx in pred_temp_idxs:
        template = templates_filtered[idx]
        try:
            rxn = rdchiralReaction(template)
            prod = rdchiralReactants(prod_smi_nomap)

            precs = rdchiralRun(rxn, prod)
            precursors.extend(precs)
        except:
            continue

    # remove duplicate predictions
    seen = []
    for prec in precursors:  # canonicalize all predictions
        prec = Chem.MolToSmiles(Chem.MolFromSmiles(prec), True)
        if prec not in seen:
            seen.append(prec)
        else:
            dup_count += 1

    if len(seen) < phase_topk:
        seen.extend(['9999'] * (phase_topk - len(seen)))
    else:
        seen = seen[:phase_topk]

    return precursors, seen, dup_count


def analyse_proposed(
    prod_smiles_phase: List[str],
    prod_smiles_mapped_phase: List[str],
    proposals_phase: Dict[str, List[str]],
):
    proposed_counter = Counter()
    total_proposed, min_proposed, max_proposed = 0, float('+inf'), float('-inf')
    key_count = 0
    for key, mapped_key in zip(prod_smiles_phase, prod_smiles_mapped_phase):
        precursors = proposals_phase[mapped_key]
        precursors_count = len(precursors)
        total_proposed += precursors_count
        if precursors_count > max_proposed:
            max_proposed = precursors_count
            prod_smi_max = key
        if precursors_count < min_proposed:
            min_proposed = precursors_count
            prod_smi_min = key

        proposed_counter[key] = precursors_count
        key_count += 1

    logging.info(f'Average precursors proposed per prod_smi (dups removed): {total_proposed / key_count}')
    logging.info(f'Min precursors: {min_proposed} for {prod_smi_min}')
    logging.info(f'Max precursors: {max_proposed} for {prod_smi_max})')

    logging.info(f'\nMost common 20:')
    for i in proposed_counter.most_common(20):
        logging.info(f'{i}')
    logging.info(f'\nLeast common 20:')
    for i in proposed_counter.most_common()[-20:]:
        logging.info(f'{i}')
    return


class NeuralSymPredictor:
    """Class for NeuralSym Predicting"""

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

        self.test_file = raw_data_files[0]

        random.seed(self.model_args.seed)
        np.random.seed(self.model_args.seed)
        torch.manual_seed(self.model_args.seed)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.templates_filtered = []
        self.init_templates()
        self.build_predict_model()

    def init_templates(self):
        templates_file = os.path.join(self.processed_data_path, "training_templates.txt")
        logging.info(f'Loading templates from file: {templates_file}')
        with open(templates_file, 'r') as f:
            templates = f.readlines()
        for p in templates:
            pa, cnt = p.strip().split(': ')
            if int(cnt) >= self.model_args.min_freq:
                self.templates_filtered.append(pa)
        logging.info(f'Total number of template patterns: {len(self.templates_filtered)}')

    def build_predict_model(self):
        if self.model_args.model_arch == 'Highway':
            self.model = TemplateNN_Highway(
                output_size=len(self.templates_filtered),
                size=self.model_args.hidden_size,
                num_layers_body=self.model_args.depth,
                input_size=self.model_args.final_fp_size
            )
        elif self.model_args.model_arch == 'FC':
            self.model = TemplateNN_FC(
                output_size=len(self.templates_filtered),
                size=self.model_args.hidden_size,
                input_size=self.model_args.fp_size
            )
        else:
            raise ValueError(f"Unrecognized model name: {self.model_args.model_arch}")

        checkpoint = torch.load(
            os.path.join(self.model_path, f"{self.data_name}.pth.tar"),
            map_location=self.device
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model = self.model.to(self.device)

        logging.info("Logging model summary")
        logging.info(self.model)
        logging.info(f"\nModel #Params: {sum([x.nelement() for x in self.model.parameters()]) / 1000} k")

    def predict(self):
        self.infer_all()
        self.compile_into_csv()

    def infer_all(self):
        """Actual file-based predicting, adapted from infer_all.py"""
        dataset = FingerprintDataset(
            os.path.join(self.processed_data_path, "to_32681_prod_fps_test.npz"),
            os.path.join(self.processed_data_path, "labels_test.npy")
        )
        loader = DataLoader(dataset, batch_size=self.model_args.bs, shuffle=False)
        del dataset

        preds = []
        loader = tqdm(loader, desc="Predicting on test")
        self.model.eval()
        with torch.no_grad():
            for data in loader:
                inputs, labels, idxs = data             # we don't need labels & idxs
                inputs = inputs.to(self.device)

                outputs = self.model(inputs)
                outputs = nn.Softmax(dim=1)(outputs)

                preds.append(torch.topk(outputs, k=self.model_args.topk, dim=1)[1])
            preds = torch.cat(preds, dim=0).squeeze(dim=-1).cpu().numpy()

        logging.info(f'preds.shape: {preds.shape}')
        np.save(os.path.join(self.test_output_path, "raw_outputs_on_test.npy"), preds)
        logging.info(f'Saved preds of test as npy!')

    def compile_into_csv(self):
        logging.info("Compiling into predictions.csv")

        preds = np.load(os.path.join(self.test_output_path, "raw_outputs_on_test.npy"))

        # load mapped_rxn_smi
        with open(self.test_file, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            clean_rxnsmi_phase = [row["reactants>reagents>production"].strip()
                                  for row in csv_reader]

        proposals_data = pd.read_csv(
            os.path.join(self.processed_data_path, "processed_test.csv"),
            index_col=None, dtype='str'
        )

        tasks = []
        for i in range(len(clean_rxnsmi_phase)):        # build tasks
            tasks.append((i, proposals_data.iloc[i, 1]))

        proposals_phase = {}
        proposed_precs_phase, prod_smiles_phase, rcts_smiles_phase = [], [], []
        proposed_precs_phase_withdups = []              # true representation of model predictions, for calc_accs()
        prod_smiles_mapped_phase = []                   # helper for analyse_proposed()
        phase_topk = self.model_args.topk
        dup_count = 0

        num_cores = self.model_args.num_cores
        logging.info(f'Parallelizing over {num_cores} cores')
        pool = multiprocessing.Pool(num_cores)

        gen_precs_partial = partial(gen_precs, self.templates_filtered, preds, phase_topk)
        for i, result in enumerate(tqdm(pool.imap(gen_precs_partial, tasks),
                                        total=len(clean_rxnsmi_phase),
                                        desc='Generating predicted reactants')):
            precursors, seen, this_dup = result
            dup_count += this_dup

            prod_smi = clean_rxnsmi_phase[i].split('>>')[-1]
            prod_smiles_mapped_phase.append(prod_smi)

            prod_smi_nomap = proposals_data.iloc[i, 1]
            prod_smiles_phase.append(prod_smi_nomap)

            rcts_smi_nomap = proposals_data.iloc[i, 2]
            rcts_smiles_phase.append(rcts_smi_nomap)

            proposals_phase[prod_smi] = precursors
            proposed_precs_phase.append(seen)
            proposed_precs_phase_withdups.append(precursors)

        pool.close()
        pool.join()

        dup_count /= len(clean_rxnsmi_phase)
        logging.info(f'Avg # duplicates per product: {dup_count}')

        analyse_proposed(
            prod_smiles_phase,
            prod_smiles_mapped_phase,
            proposals_phase,        # this func needs this to be a dict {mapped_prod_smi: proposals}
        )

        zipped = []
        for rxn_smi, prod_smi, rcts_smi, proposed_rcts_smi in zip(
                clean_rxnsmi_phase,
                prod_smiles_phase,
                rcts_smiles_phase,
                proposed_precs_phase,
        ):
            result = [prod_smi]
            result.extend(proposed_rcts_smi)
            zipped.append(result)

        logging.info('Zipped all info for each rxn_smi into a list for dataframe creation!')

        temp_dataframe = pd.DataFrame(data={'zipped': zipped})
        phase_dataframe = pd.DataFrame(
            temp_dataframe['zipped'].to_list(),
            index=temp_dataframe.index
        )

        proposed_col_names = [f'cand_precursor_{i}' for i in range(1, self.model_args.topk + 1)]
        col_names = ['prod_smi']
        col_names.extend(proposed_col_names)
        phase_dataframe.columns = col_names

        phase_dataframe.to_csv(os.path.join(self.test_output_path, "predictions.csv"), index=False)
        logging.info(f'Saved proposals of test as CSV!')
