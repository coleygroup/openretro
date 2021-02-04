import csv
import logging
import multiprocessing
import numpy as np
import os
import pandas as pd
import pickle
from base.processor_base import Processor
from models.retroxpert_model.extract_semi_template_pattern import cano_smarts, get_tpl
from models.retroxpert_model.preprocessing import get_atom_features, get_bond_features, \
    get_atomidx2mapidx, get_mapidx2atomidx, get_order, get_smarts_pieces
from rdkit import Chem
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
        self.check_count = 100
        self.train_file, self.val_file, self.test_file = raw_data_files
        self.num_cores = num_cores

    def check_data_format(self) -> None:
        """Check that all files exists and the data format is correct for all"""
        super().check_data_format()

        logging.info(f"Checking the first {self.check_count} entries for each file")
        for fn in self.raw_data_files:
            if not fn:
                continue

            with open(fn, "r") as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for i, row in enumerate(csv_reader):
                    if i == 0:                          # header
                        continue
                    if i > self.check_count:            # check the first few rows
                        break

                    assert "rxn_smiles" in row, \
                        f"Error processing file {fn} line {i}, ensure column 'rxn_smiles' is included!"
                    if self.model_args.typed:
                        assert "class" in row and (row["class"].isnumeric() or row["class"] == "UNK"), \
                            f"Error processing file {fn} line {i}, " \
                            f"if --typed is specified, ensure 'class' column is numeric or 'UNK'"

                    reactants, products = row["rxn_smiles"].split(">>")
                    Chem.MolFromSmiles(reactants)       # simply ensures that SMILES can be parsed
                    Chem.MolFromSmiles(products)        # simply ensures that SMILES can be parsed

        logging.info("Data format check passed")

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
            reaction_list = df["rxn_smiles"]
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
                    for atom, mat in zip(mol_cano.GetAtoms(), matches[0]):
                        atom.SetAtomMapNum(index2mapnums[mat])
                    product = Chem.MolToSmiles(mol_cano, canonical=False)
                reaction_list_new.append(f"{reactant}>>{product}")

            df["rxn_smiles"] = reaction_list_new
            output_csv_file = os.path.join(self.processed_data_path, f"canonicalized_{phase}.csv")
            df.to_csv(output_csv_file, index=False)

    def preprocess_core(self):
        """Core of stage 1 preprocessing, adapted from preprocessing.py"""
        logging.info(f"Running preprocessing core for stage 1")

        opennmt_data_path = os.path.join(self.processed_data_path, "opennmt_data")
        os.makedirs(opennmt_data_path, exist_ok=True)

        for phase in ["train", "val", "test"]:
            csv_file = os.path.join(self.processed_data_path, f"canonicalized_{phase}.csv")
            df = pd.read_csv(csv_file)

            rxn_data_dict = {}
            of_src = open(os.path.join(opennmt_data_path, f"src-{phase}.txt"), "w")
            of_tgt = open(os.path.join(opennmt_data_path, f"tgt-{phase}.txt"), "w")

            if phase == "train":
                of_src_aug = open(os.path.join(opennmt_data_path, f"src-train-aug.txt"), "w")
                of_tgt_aug = open(os.path.join(opennmt_data_path, f"tgt-train-aug.txt"), "w")

            for i, row in tqdm(df.iterrows()):
                # adapted from __main__()
                reactant, product = row["rxn_smiles"].split(">>")
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
                src_item, tgt_item = get_smarts_pieces(product_mol, product_adj, target_adj, reactants)
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

            ofn = os.path.join(self.processed_data_path, f"rxn_data_{phase}.pkl")
            with open(ofn, "wb") as of:
                pickle.dump(rxn_data_dict, of)

    def extract_semi_templates(self):
        """Adapted from extract_semi_template_pattern.py"""
        logging.info("Extracting semi-templates")

        pattern_file = os.path.join(self.processed_data_path, "product_patterns.txt")
        rxn_data_file = os.path.join(self.processed_data_path, f"rxn_data_train.pkl")
        with open(rxn_data_file, "rb") as f:
            rxn_data_dict = pickle.load(f)

        patterns = {}
        rxns = []
        for i, rxn_data in rxn_data_dict.items():
            reactant = Chem.MolToSmiles(rxn_data["reactant_mol"], canonical=False)
            product = Chem.MolToSmiles(rxn_data["product_mol"], canonical=False)
            rxns.append((i, reactant, product))
        logging.info(f"Total training reactions: {len(rxns)}")

        pool = multiprocessing.Pool(self.num_cores)
        for result in tqdm(pool.imap_unordered(get_tpl, rxns), total=len(rxns)):
            idx, template = result
            if "reaction_smarts" not in template:
                continue
            product_pattern = cano_smarts(template["products"])
            if product_pattern not in patterns:
                patterns[product_pattern] = 1
            else:
                patterns[product_pattern] += 1

        patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        patterns = [f"{p[0]}: {p[1]}\n" for p in patterns]
        logging.info(f"Total patterns: {len(patterns)}")
        with open(pattern_file, "w") as f:
            f.writelines(patterns)

        pool.close()
        pool.join()

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

        # ugly solution -- passing list for pool using global variables
        global G_patterns_filtered
        G_patterns_filtered = patterns_filtered

        for phase in ["train", "val", "test"]:
            rxn_data_file = os.path.join(self.processed_data_path, f"rxn_data_{phase}.pkl")
            with open(rxn_data_file, "rb") as f:
                rxn_data_dict = pickle.load(f)

            tasks = [(idx, Chem.MolToSmiles(rxn_data["product_mol"], canonical=False))
                     for idx, rxn_data in rxn_data_dict.items()]

            counter = []
            pool = multiprocessing.Pool(self.num_cores)
            for result in tqdm(pool.imap_unordered(find_all_patterns, tasks), total=len(tasks)):
                k, pattern_feature = result
                reaction_data = rxn_data_dict[k]
                reaction_data["pattern_feat"] = pattern_feature.astype(np.bool)

                pa = np.sum(pattern_feature, axis=0)            # what's this?
                counter.append(np.sum(pa > 0))

            logging.info(f"# ave center per mol: {np.mean(counter)}")
            with open(rxn_data_file, "wb") as of:
                pickle.dump(rxn_data_dict, of)

            pool.close()
            pool.join()
