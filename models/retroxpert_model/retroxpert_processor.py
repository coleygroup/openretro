import csv
import json
import logging
import multiprocessing
import numpy as np
import os
import pandas as pd
import pickle
import time
from base.processor_base import Processor
from collections import Counter
from concurrent.futures import TimeoutError
from models.retroxpert_model.data import RetroCenterDatasets
from models.retroxpert_model.extract_semi_template_pattern import cano_smarts, get_tpl
from models.retroxpert_model.preprocessing import get_atom_features, get_bond_features, \
    get_atomidx2mapidx, get_idx, get_mapidx2atomidx, get_order, smarts2smiles, smi_tokenizer, get_smarts_pieces_s2
from models.retroxpert_model.preprocessing import get_smarts_pieces as get_smarts_pieces_s1
from models.retroxpert_model.retroxpert_trainer import collate, RetroXpertTrainerS1
from onmt.bin.preprocess import preprocess as onmt_preprocess
from pebble import ProcessPool
from rdkit import Chem
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List


G_patterns_filtered = []


def find_all_patterns(task):
    global G_patterns_filtered

    k, product = task
    product_mol = Chem.MolFromSmiles(product)
    [a.SetAtomMapNum(0) for a in product_mol.GetAtoms()]
    matches_all = {}
    for idx, pattern in enumerate(G_patterns_filtered):
        pattern_mol = Chem.MolFromSmarts(pattern)
        if pattern_mol is None:
            logging.info(f"error: pattern_mol is None, idx: {idx}")
        try:
            matches = product_mol.GetSubstructMatches(pattern_mol,
                                                      useChirality=False)
        except:
            logging.info(f"Caught some exception at {k}th task. Continue anyway.")
            continue
        else:
            if len(matches) > 0 and len(matches[0]) > 0:
                matches_all[idx] = matches
    if len(matches_all) == 0:
        print(product)
    num_atoms = product_mol.GetNumAtoms()
    pattern_feature = np.zeros((len(G_patterns_filtered), num_atoms))
    for idx, matches in matches_all.items():
        if len(matches) > 1 and isinstance(matches[0], tuple):
            for match in matches:
                np.put(pattern_feature[idx], match, 1)
        else:
            np.put(pattern_feature[idx], matches, 1)
    pattern_feature = pattern_feature.transpose().astype('bool_')
    return k, pattern_feature


class RetroXpertProcessorS1(Processor):
    """Class for RetroXpert Preprocessing, Stage 1"""

    def __init__(self,
                 model_name: str,
                 model_args,
                 model_config: Dict[str, any],
                 data_name: str,
                 raw_data_files: List[str],
                 processed_data_path: str,
                 num_cores: int = None):
        super().__init__(model_name=model_name,
                         model_args=model_args,
                         model_config=model_config,
                         data_name=data_name,
                         raw_data_files=raw_data_files,
                         processed_data_path=processed_data_path)
        self.train_file, self.val_file, self.test_file = raw_data_files
        self.num_cores = num_cores

    def preprocess(self) -> None:
        """Actual file-based preprocessing"""
        self.canonicalize_products()
        self.preprocess_core()
        self.extract_semi_templates()
        self.find_patterns_in_data()

    def canonicalize_products(self):
        """Adapted from canonicalize_products.py"""
        logging.info(f"Canonicalizing products")

        for phase, csv_file in [("train", self.train_file),
                                ("val", self.val_file),
                                ("test", self.test_file)]:
            df = pd.read_csv(csv_file)
            output_csv_file = os.path.join(self.processed_data_path, f"canonicalized_{phase}.csv")

            if os.path.exists(output_csv_file):
                logging.info(f"Output file found at {output_csv_file}, skipping preprocessing core for phase {phase}")
                continue

            reaction_list = df["reactants>reagents>production"]
            reaction_list_new = []

            for reaction in tqdm(reaction_list):
                reactant, product = reaction.split(">>")
                mol = Chem.MolFromSmiles(product)
                index2mapnums = {}
                for atom in mol.GetAtoms():
                    index2mapnums[atom.GetIdx()] = atom.GetAtomMapNum()

                # canonicalize the product smiles
                mol_cano = Chem.RWMol(mol)
                [atom.SetAtomMapNum(0) for atom in mol_cano.GetAtoms()]
                smi_cano = Chem.MolToSmiles(mol_cano)
                mol_cano = Chem.MolFromSmiles(smi_cano)

                matches = mol.GetSubstructMatches(mol_cano)
                if matches:
                    mapnums_old2new = {}
                    for atom, mat in zip(mol_cano.GetAtoms(), matches[0]):
                        mapnums_old2new[index2mapnums[mat]] = 1 + atom.GetIdx()
                        # update product mapping numbers according to canonical atom order
                        # to completely remove potential information leak
                        atom.SetAtomMapNum(1 + atom.GetIdx())
                    product = Chem.MolToSmiles(mol_cano)
                    # update reactant mapping numbers accordingly
                    mol_react = Chem.MolFromSmiles(reactant)
                    for atom in mol_react.GetAtoms():
                        if atom.GetAtomMapNum() > 0:
                            atom.SetAtomMapNum(mapnums_old2new[atom.GetAtomMapNum()])
                    reactant = Chem.MolToSmiles(mol_react)
                reaction_list_new.append(f"{reactant}>>{product}")

            df["reactants>reagents>production"] = reaction_list_new
            df.to_csv(output_csv_file, index=False)

    def preprocess_core(self):
        """Core of stage 1 preprocessing, adapted from preprocessing.py"""
        logging.info(f"Running preprocessing core for stage 1")

        opennmt_data_path = os.path.join(self.processed_data_path, "opennmt_data_s1")
        os.makedirs(opennmt_data_path, exist_ok=True)

        for phase in ["train", "val", "test"]:
            csv_file = os.path.join(self.processed_data_path, f"canonicalized_{phase}.csv")
            ofn = os.path.join(self.processed_data_path, f"rxn_data_{phase}.pkl")

            if os.path.exists(ofn):
                logging.info(f"Output file found at {ofn}, skipping preprocessing core for phase {phase}")
                continue

            df = pd.read_csv(csv_file)

            rxn_data_dict = {}
            of_src = open(os.path.join(opennmt_data_path, f"src-{phase}.txt"), "w")
            of_tgt = open(os.path.join(opennmt_data_path, f"tgt-{phase}.txt"), "w")

            if phase == "train":
                of_src_aug = open(os.path.join(opennmt_data_path, f"src-train-aug.txt"), "w")
                of_tgt_aug = open(os.path.join(opennmt_data_path, f"tgt-train-aug.txt"), "w")

            for i, row in tqdm(df.iterrows()):
                # adapted from __main__()
                reactant, product = row["reactants>reagents>production"].split(">>")
                reaction_class = int(row["class"] - 1) if self.model_args.typed else "UNK"

                product_mol = Chem.MolFromSmiles(product)
                reactant_mol = Chem.MolFromSmiles(reactant)

                product_adj = Chem.rdmolops.GetAdjacencyMatrix(product_mol)
                product_adj = product_adj + np.eye(product_adj.shape[0])
                product_adj = product_adj.astype(np.bool)
                reactant_adj = Chem.rdmolops.GetAdjacencyMatrix(reactant_mol)
                reactant_adj = reactant_adj + np.eye(reactant_adj.shape[0])
                reactant_adj = reactant_adj.astype(np.bool)

                patomidx2pmapidx = get_atomidx2mapidx(product_mol)
                rmapidx2ratomidx = get_mapidx2atomidx(reactant_mol)

                order = get_order(product_mol, patomidx2pmapidx, rmapidx2ratomidx)
                target_adj = reactant_adj[order][:, order]

                product_bond_features = get_bond_features(product_mol)
                product_atom_features = get_atom_features(product_mol)

                rxn_data = {
                    'rxn_type': reaction_class,
                    'product_adj': product_adj,
                    'product_mol': product_mol,
                    'product_bond_features': product_bond_features,
                    'product_atom_features': product_atom_features,
                    'target_adj': target_adj,
                    'reactant_mol': reactant_mol
                }
                rxn_data_dict[i] = rxn_data

                # adapted from generate_opennmt_data()
                reactants = reactant.split(".")
                src_item, tgt_item = get_smarts_pieces_s1(product_mol, product_adj, target_adj, reactants)
                of_src.write(f"{i} [RXN_{reaction_class}] {product} [PREDICT] {src_item}\n")
                of_tgt.write(f"{tgt_item}\n")

                # data augmentation fro training data
                if phase == "train":
                    of_src_aug.write(f"{i} [RXN_{reaction_class}] {product} [PREDICT] {src_item}\n")
                    of_tgt_aug.write(f"{tgt_item}\n")

                    reactants = tgt_item.split(".")
                    if len(reactants) == 1:
                        continue
                    synthons = src_item.strip().split(".")
                    src_item = " . ".join(synthons[::-1]).strip()           # .reverse() is an in-place op
                    tgt_item = " . ".join(reactants[::-1]).strip()

                    of_src_aug.write(f"{i} [RXN_{reaction_class}] {product} [PREDICT] {src_item}\n")
                    of_tgt_aug.write(f"{tgt_item}\n")

            logging.info(f"Data size: {i + 1}")
            of_src.close()
            of_tgt.close()

            if phase == "train":
                of_src_aug.close()
                of_tgt_aug.close()

            with open(ofn, "wb") as of:
                pickle.dump(rxn_data_dict, of)

    def extract_semi_templates(self):
        """Adapted from extract_semi_template_pattern.py"""
        logging.info("Extracting semi-templates")

        pattern_file = os.path.join(self.processed_data_path, "product_patterns.txt")
        rxn_data_file = os.path.join(self.processed_data_path, "rxn_data_train.pkl")

        if os.path.exists(pattern_file):
            logging.info(f"Output file found at {pattern_file}, skipping extract_semi_templates()")
            return

        with open(rxn_data_file, "rb") as f:
            rxn_data_dict = pickle.load(f)

        patterns = {}
        rxns = []
        for i, rxn_data in tqdm(rxn_data_dict.items()):
            reactant = Chem.MolToSmiles(rxn_data["reactant_mol"], canonical=False)
            product = Chem.MolToSmiles(rxn_data["product_mol"], canonical=False)
            rxns.append((i, reactant, product))
        logging.info(f"Total training reactions: {len(rxns)}")
        del rxn_data_dict

        with ProcessPool(max_workers=self.num_cores) as pool:
            # had to resort to pebble to add timeout. rdchiral could hang
            future = pool.map(get_tpl, rxns, timeout=10)

            iterator = future.result()
            while True:
                try:
                    result = next(iterator)
                    idx, product_pattern = result

                    if product_pattern is None:
                        continue
                    if product_pattern not in patterns:
                        patterns[product_pattern] = 1
                    else:
                        patterns[product_pattern] += 1
                except StopIteration:
                    break
                except TimeoutError as error:
                    logging.info(f"get_tpl call took more than {error.args} seconds")

        logging.info(f"Finished getting templates. Creating patterns dict")
        del rxns, future

        logging.info("Sorting all patterns")
        patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        # patterns = [f"{p[0]}: {p[1]}\n" for p in patterns]
        logging.info(f"Sorted. Total patterns: {len(patterns)}")
        with open(pattern_file, "w") as of:
            for pattern in patterns:
                of.write(f"{pattern[0]}: {pattern[1]}\n")

    def find_patterns_in_data(self):
        """Adapted from extract_semi_template_pattern.py"""
        logging.info("Finding patterns in all data")

        patterns_filtered = []
        pattern_file = os.path.join(self.processed_data_path, "product_patterns.txt")
        with open(pattern_file, "r") as f:
            for line in f:
                pattern, count = line.strip().split(": ")
                if int(count.strip()) >= self.model_args.min_freq:
                    patterns_filtered.append(pattern)
        logging.info(f"Filtered patterns by min frequency {self.model_args.min_freq}, "
                     f"remaining pattern count: {len(patterns_filtered)}")

        metadata_file = os.path.join(self.processed_data_path, "metadata.json")
        with open(metadata_file, "w") as of:
            json.dump({"semi_template_count": len(patterns_filtered)}, of, ensure_ascii=True, indent=4)

        # ugly solution -- passing list for pool using global variables
        global G_patterns_filtered
        G_patterns_filtered = patterns_filtered

        for phase in ["train", "val", "test"]:
            rxn_data_file = os.path.join(self.processed_data_path, f"rxn_data_{phase}.pkl")
            pattern_feat_file = os.path.join(self.processed_data_path, f"pattern_feat_{phase}.npz")

            logging.info(f"Loading from {rxn_data_file}")
            with open(rxn_data_file, "rb") as f:
                rxn_data_dict = pickle.load(f)

            tasks = [(idx, Chem.MolToSmiles(rxn_data["product_mol"], canonical=False))
                     for idx, rxn_data in rxn_data_dict.items()]

            counter = []
            pattern_features = []
            pattern_features_lens = []

            pool = multiprocessing.Pool(self.num_cores)
            for i, result in enumerate(tqdm(pool.imap(find_all_patterns, tasks), total=len(tasks))):
                k, pattern_feature = result
                assert i == k

                pattern_features.append(pattern_feature)
                pattern_features_lens.append(pattern_feature.shape[0])

                pa = np.sum(pattern_feature, axis=0)            # what's this?
                counter.append(np.sum(pa > 0))

            pattern_features = np.concatenate(pattern_features, axis=0)
            pattern_features_lens = np.asarray(pattern_features_lens)

            logging.info(f"# ave center per mol: {np.mean(counter)}, "
                         f"shape of pattern_features: {pattern_features.shape}, "
                         f"shape of pattern_features_lens: {pattern_features_lens.shape}, "
                         f"dumping into {pattern_feat_file}")
            np.savez(pattern_feat_file,
                     pattern_features=pattern_features,
                     pattern_features_lens=pattern_features_lens)
            logging.info("Dumped. Cleaning up")

            pool.close()
            pool.join()

            del pattern_features, pattern_features_lens


class RetroXpertProcessorS2(Processor):
    """Class for RetroXpert Preprocessing, Stage 2"""

    def __init__(self,
                 model_name: str,
                 model_args,
                 model_config: Dict[str, any],
                 data_name: str,
                 raw_data_files: List[str],
                 processed_data_path: str,
                 num_cores: int = None):
        super().__init__(model_name=model_name,
                         model_args=model_args,
                         model_config=model_config,
                         data_name=data_name,
                         raw_data_files=raw_data_files,
                         processed_data_path=processed_data_path)
        self.check_count = 100

        self.trainer_s1 = RetroXpertTrainerS1(
            model_name=model_name,
            model_args=model_args,
            model_config=model_config,
            data_name=data_name,
            raw_data_files=raw_data_files,
            processed_data_path=processed_data_path,
            model_path=model_args.model_path_s1
        )
        self.trainer_s1.build_train_model()

    def check_data_format(self) -> None:
        logging.info("For RetroXpert Stage 2, there is no additional raw data file needed")
        logging.info("Data format check passed (trivially)")

    def preprocess(self) -> None:
        """Actual file-based preprocessing"""
        self.test_and_save(data_split="test")
        self.test_and_save(data_split="train")
        self.generate_formatted_dataset()
        self.prepare_test_prediction()
        self.prepare_train_error_aug()
        self.onmt_preprocess()

    def test_and_save(self, data_split: str):
        fn = f"rxn_data_{data_split}.pkl"
        fn_pattern = f"pattern_feat_{data_split}.npz"

        logging.info(f"Testing on {fn} to generate stage 1 results on {data_split}")

        disconnection_fn = os.path.join(
            self.processed_data_path, f"{data_split}_disconnection_{self.trainer_s1.exp_name}.txt")
        result_fn = os.path.join(
            self.processed_data_path, f"{data_split}_result_{self.trainer_s1.exp_name}.txt")
        result_mol_fn = os.path.join(
            self.processed_data_path, f"{data_split}_result_mol_{self.trainer_s1.exp_name}.txt")

        if all(os.path.exists(fn) for fn in [disconnection_fn, result_fn, result_mol_fn]):
            logging.info(f"Found test results for {data_split}, skipping testing")
            return

        data = RetroCenterDatasets(processed_data_path=self.processed_data_path,
                                   fns=[fn, fn_pattern])
        dataloader = DataLoader(data,
                                batch_size=4 * self.model_args.batch_size,
                                shuffle=False,
                                num_workers=0,
                                collate_fn=collate)
        self.trainer_s1.test(dataloader=dataloader,
                             data_split=data_split,
                             save_pred=True)

    def generate_formatted_dataset(self):
        """Adapted from prepare_data.py"""
        logging.info("Generating formatted dataset for OpenNMT (Stage 2)")

        src = {
            'train': 'src-train-aug.txt',
            'test': 'src-test.txt',
            'val': 'src-val.txt',
        }
        tgt = {
            'train': 'tgt-train-aug.txt',
            'test': 'tgt-test.txt',
            'val': 'tgt-val.txt',
        }

        savedir = os.path.join(self.processed_data_path, "opennmt_data_s2")
        os.makedirs(savedir, exist_ok=True)

        tokens = Counter()
        for data_set in ['val', 'train', 'test']:
            with open(os.path.join(self.processed_data_path, "opennmt_data_s1", src[data_set])) as f:
                srcs = f.readlines()
            with open(os.path.join(self.processed_data_path, "opennmt_data_s1", tgt[data_set])) as f:
                tgts = f.readlines()

            src_lines = []
            tgt_lines = []
            for s, t in tqdm(list(zip(srcs, tgts))):
                tgt_items = t.strip().split()
                src_items = s.strip().split()
                src_items[2] = smi_tokenizer(smarts2smiles(src_items[2]))
                tokens.update(src_items[2].split(' '))
                for idx in range(4, len(src_items)):
                    if src_items[idx] == '.':
                        continue
                    src_items[idx] = smi_tokenizer(smarts2smiles(src_items[idx], canonical=False))
                    tokens.update(src_items[idx].split(' '))
                for idx in range(len(tgt_items)):
                    if tgt_items[idx] == '.':
                        continue
                    tgt_items[idx] = smi_tokenizer(smarts2smiles(tgt_items[idx]))
                    tokens.update(tgt_items[idx].split(' '))

                if not self.model_args.typed:
                    src_items[1] = '[RXN_0]'

                src_line = ' '.join(src_items[1:])
                tgt_line = ' '.join(tgt_items)
                src_lines.append(src_line + '\n')
                tgt_lines.append(tgt_line + '\n')

            src_file = os.path.join(savedir, src[data_set])
            logging.info(f"src_file: {src_file}")
            with open(src_file, "w") as f:
                f.writelines(src_lines)

            tgt_file = os.path.join(savedir, tgt[data_set])
            logging.info(f"tgt_file: {tgt_file}")
            with open(tgt_file, "w") as f:
                f.writelines(tgt_lines)

    @staticmethod
    def get_bond_disconnection_prediction(
            pred_results_file: str, bond_pred_results_file: str, reaction_data_file: str):
        with open(pred_results_file) as f:
            pred_results = f.readlines()
        with open(bond_pred_results_file) as f:
            bond_pred_results = f.readlines()
        with open(reaction_data_file, "rb") as f:
            rxn_data_dict = pickle.load(f)

        product_adjs = []
        product_mols = []
        product_smiles = []
        for i, rxn_data in rxn_data_dict.items():
            product_adjs.append(rxn_data["product_adj"])
            product_mols.append(rxn_data["product_mol"])
            product_smiles.append(Chem.MolToSmiles(rxn_data["product_mol"], canonical=False))

        assert len(product_smiles) == len(bond_pred_results)

        cnt = 0
        guided_pred_results = []
        bond_disconnection = []
        bond_disconnection_gt = []
        for i in range(len(bond_pred_results)):
            bond_pred_items = bond_pred_results[i].strip().split()
            bond_change_num = int(bond_pred_items[1]) * 2
            bond_change_num_gt = int(bond_pred_items[0]) * 2

            gt_adj_list = pred_results[3 * i + 1].strip().split(" ")
            gt_adj_list = [int(k) for k in gt_adj_list]
            gt_adj_index = np.argsort(gt_adj_list)
            gt_adj_index = gt_adj_index[:bond_change_num_gt]

            pred_adj_list = pred_results[3 * i + 2].strip().split(" ")
            pred_adj_list = [float(k) for k in pred_adj_list]
            pred_adj_index = np.argsort(pred_adj_list)
            pred_adj_index = pred_adj_index[:bond_change_num]

            bond_disconnection.append(pred_adj_index)
            bond_disconnection_gt.append(gt_adj_index)
            res = set(gt_adj_index) == set(pred_adj_index)
            guided_pred_results.append(int(res))
            cnt += res

        logging.info(f"guided bond_disconnection prediction cnt and acc: {cnt} {cnt / len(bond_pred_results)}")
        logging.info(f"bond_disconnection len: {len(bond_disconnection)}")

        return product_adjs, product_mols, bond_disconnection, guided_pred_results

    def prepare_test_prediction(self):
        """Adapted from prepare_test_prediction.py"""
        logging.info("Using bond disconnection prediction to generate synthons for test data")

        pred_results_file = os.path.join(
            self.processed_data_path, f"test_result_mol_{self.trainer_s1.exp_name}.txt")
        bond_pred_results_file = os.path.join(
            self.processed_data_path, f"test_disconnection_{self.trainer_s1.exp_name}.txt")
        reaction_data_file = os.path.join(
            self.processed_data_path, f"rxn_data_test.pkl")

        product_adjs, product_mols, bond_disconnection, guided_pred_results = self.get_bond_disconnection_prediction(
            pred_results_file=pred_results_file,
            bond_pred_results_file=bond_pred_results_file,
            reaction_data_file=reaction_data_file
        )

        logging.info("Generate synthons from bond disconnection prediction")
        synthons = []
        for i, prod_adj in enumerate(product_adjs):
            x_adj = np.array(prod_adj)
            # find 1 index
            idxes = np.argwhere(x_adj > 0)
            pred_adj = prod_adj.copy()
            for k in bond_disconnection[i]:
                idx = idxes[k]
                assert pred_adj[idx[0], idx[1]] == 1
                pred_adj[idx[0], idx[1]] = 0

            pred_synthon = get_smarts_pieces_s2(product_mols[i], prod_adj, pred_adj)
            synthons.append(pred_synthon)

        with open(os.path.join(self.processed_data_path, "opennmt_data_s1", "src-test.txt")) as f:
            srcs = f.readlines()
        assert len(synthons) == len(srcs)

        savedir = os.path.join(self.processed_data_path, "opennmt_data_s2")
        src_test_prediction = os.path.join(savedir, "src-test-prediction.txt")
        logging.info(f"Saving src_test_prediction to {src_test_prediction}")

        cnt = 0
        with open(src_test_prediction, "w") as f:
            for src, synthon in zip(srcs, synthons):
                src_items = src.split(" ")
                src_items[2] = smi_tokenizer(smarts2smiles(src_items[2]))
                if not self.model_args.typed:
                    src_items[1] = '[RXN_0]'

                syns = synthon.split(".")
                syns = [smarts2smiles(s, canonical=False) for s in syns]

                # Double check the synthon prediction accuracy
                syns_gt = [smarts2smiles(s, canonical=False) for s in src_items[4:] if s != "."]
                cnt += set(syns_gt) == set(syns)

                syns = [smi_tokenizer(s) for s in syns]
                src_line = ' '.join(src_items[1:4]) + ' ' + ' . '.join(syns) + '\n'
                f.write(src_line)

        logging.info(f"double checking guided synthon prediction acc: {cnt / len(synthons)}")

    def prepare_train_error_aug(self):
        """Adapted from prepare_train_error_aug.py"""
        logging.info("Using erroneous bond disconnection prediction to augment train data")

        pred_results_file = os.path.join(
            self.processed_data_path, f"train_result_mol_{self.trainer_s1.exp_name}.txt")
        bond_pred_results_file = os.path.join(
            self.processed_data_path, f"train_disconnection_{self.trainer_s1.exp_name}.txt")
        reaction_data_file = os.path.join(
            self.processed_data_path, f"rxn_data_train.pkl")

        product_adjs, product_mols, bond_disconnection, guided_pred_results = self.get_bond_disconnection_prediction(
            pred_results_file=pred_results_file,
            bond_pred_results_file=bond_pred_results_file,
            reaction_data_file=reaction_data_file
        )

        with open(os.path.join(self.processed_data_path, "opennmt_data_s1", "src-train.txt")) as f:
            srcs = f.readlines()
        with open(os.path.join(self.processed_data_path, "opennmt_data_s1", "tgt-train.txt")) as f:
            tgts = f.readlines()

        # Generate synthons from bond disconnection prediction
        sources = []
        targets = []
        for i, prod_adj in enumerate(product_adjs):
            if guided_pred_results[i] == 1:
                continue
            x_adj = np.array(prod_adj)
            # find 1 index
            idxes = np.argwhere(x_adj > 0)
            pred_adj = prod_adj.copy()
            for k in bond_disconnection[i]:
                idx = idxes[k]
                assert pred_adj[idx[0], idx[1]] == 1
                pred_adj[idx[0], idx[1]] = 0

            pred_synthon = get_smarts_pieces_s2(product_mols[i], prod_adj, pred_adj)

            reactants = tgts[i].split('.')
            reactants = [r.strip() for r in reactants]

            syn_idx_list = [get_idx(s) for s in pred_synthon.split('.')]
            react_idx_list = [get_idx(r) for r in reactants]
            react_max_common_synthon_index = []
            for react_idx in react_idx_list:
                react_common_idx_cnt = []
                for syn_idx in syn_idx_list:
                    common_idx = list(set(syn_idx) & set(react_idx))
                    react_common_idx_cnt.append(len(common_idx))
                max_cnt = max(react_common_idx_cnt)
                react_max_common_index = react_common_idx_cnt.index(max_cnt)
                react_max_common_synthon_index.append(react_max_common_index)
            react_synthon_index = np.argsort(react_max_common_synthon_index).tolist()
            reactants = [reactants[k] for k in react_synthon_index]

            # remove mapping number
            syns = pred_synthon.split('.')
            syns = [smarts2smiles(s, canonical=False) for s in syns]
            syns = [smi_tokenizer(s) for s in syns]
            src_items = srcs[i].strip().split(' ')
            src_items[2] = smi_tokenizer(smarts2smiles(src_items[2]))
            if not self.model_args.typed:
                src_items[1] = '[RXN_0]'
            src_line = ' '.join(src_items[1:4]) + ' ' + ' . '.join(syns) + '\n'

            reactants = [smi_tokenizer(smarts2smiles(r)) for r in reactants]
            tgt_line = ' . '.join(reactants) + '\n'

            sources.append(src_line)
            targets.append(tgt_line)

        logging.info(f"augmentation data size: {len(sources)}")

        savedir = os.path.join(self.processed_data_path, "opennmt_data_s2")
        with open(os.path.join(savedir, "src-train-aug.txt")) as f:
            srcs = f.readlines()
        with open(os.path.join(savedir, "tgt-train-aug.txt")) as f:
            tgts = f.readlines()

        src_train_aug_err = os.path.join(savedir, "src-train-aug-err.txt")
        logging.info(f"Saving src_train_aug_err to {src_train_aug_err}")
        with open(src_train_aug_err, "w") as f:
            f.writelines(srcs + sources)

        tgt_train_aug_err = os.path.join(savedir, "tgt-train-aug-err.txt")
        logging.info(f"Saving tgt_train_aug_err to {tgt_train_aug_err}")
        with open(tgt_train_aug_err, "w") as f:
            f.writelines(tgts + targets)

    def onmt_preprocess(self):
        logging.info("Running onmt preprocessing")

        logging.info("Overwriting model args, (hardcoding essentially)")
        self.overwrite_model_args()
        logging.info(f"Updated model args: {self.model_args}")

        onmt_preprocess(self.model_args)

    def overwrite_model_args(self):
        """Overwrite model args"""
        # Paths
        savedir = os.path.join(self.processed_data_path, "opennmt_data_s2")
        self.model_args.save_data = os.path.join(self.processed_data_path, "bin")
        self.model_args.train_src = [os.path.join(savedir, f"src-train-aug-err.txt")]
        self.model_args.train_tgt = [os.path.join(savedir, f"tgt-train-aug-err.txt")]
        self.model_args.valid_src = os.path.join(savedir, f"src-val.txt")
        self.model_args.valid_tgt = os.path.join(savedir, f"tgt-val.txt")
        # Runtime args, adapted from OpenNMT-pt/script/{DATASET}/preprocess.sh
        self.model_args.src_seq_length = 1000
        self.model_args.tgt_seq_length = 1000
        self.model_args.src_vocab_size = 1000
        self.model_args.tgt_vocab_size = 1000
        self.model_args.overwrite = True
        self.model_args.share_vocab = True
        self.model_args.subword_prefix = "ThisIsAHardCode"  # an arg for BART, leading to weird logging error
