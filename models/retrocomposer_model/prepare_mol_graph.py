
import argparse
import itertools
import json
import logging
import numpy as np
import random
import os
import torch
import multiprocessing

from collections import Counter
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.data import Batch
from tqdm import tqdm

from models.retrocomposer_model.chemutils import cano_smarts, cano_smarts_, cano_smiles, cano_smiles_
from models.retrocomposer_model.extract_templates import Reactor
from models.retrocomposer_model.process_templates import compose_tmpl


def get_onehot(item, item_list):
    return list(map(lambda s: item == s, item_list))

def get_symbol_onehot(symbol):
    symbol_list = ['O', 'N', 'Si', 'I', 'C', 'Br', 'Sn', 'Mg', 'Cu', 'S', 'P', 'Se', 'F', 'B', 'Cl', 'Zn', 'unk']
    if symbol not in symbol_list:
        symbol = 'unk'
    return list(map(lambda s: symbol == s, symbol_list))

def get_atom_feature(atom):
    degree_onehot = get_onehot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    H_num_onehot = get_onehot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    formal_charge = get_onehot(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
    chiral_tag = get_onehot(int(atom.GetChiralTag()), [0, 1, 2, 3])
    hybridization = get_onehot(
        atom.GetHybridization(),
        [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ]
    )
    symbol_onehot = get_symbol_onehot(atom.GetSymbol())
    # Atom mass scaled to about the same range as other features
    atom_feature = degree_onehot + H_num_onehot + formal_charge + chiral_tag + hybridization + [atom.GetIsAromatic()] + [atom.GetMass() * 0.01] + symbol_onehot

    return atom_feature

def get_bond_features(bond):
    """
    Builds a feature vector for a bond.
    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    bt = bond.GetBondType()
    fbond = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        (bond.GetIsConjugated() if bt is not None else 0),
        (bond.IsInRing() if bt is not None else 0)
    ]
    fbond += get_onehot(int(bond.GetStereo()), list(range(6)))
    return fbond

def mol_to_graph_data_obj(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(get_atom_feature(atom))
    x = torch.tensor(np.array(atom_features_list), dtype=torch.float32)

    # bonds
    num_bond_features = 12   # bond type, bond direction
    edges_list = []
    edge_features_list = []
    if len(mol.GetBonds()) > 0: # mol has bonds
        for bk, bond in enumerate(mol.GetBonds()):
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = get_bond_features(bond)
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.bool)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.bool)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.atom_len = len(mol.GetAtoms())
    data.edge_len = len(mol.GetBonds())

    return data


class MoleculeDataset(Dataset):
    def __init__(self, root, split='train', load_mol=False, min_k_prod=0, min_k_react=0):
        """
        Adapted from https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/loader.py
        :param root: directory of the dataset, containing a raw and processed dir.
            The raw dir should contain the file containing the smiles, and the
            processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for USPTO50K
        """
        self.split = split
        self.root = os.path.join(root)
        super(MoleculeDataset, self).__init__(self.root)
        self.seq_to_templates_file = os.path.join(root, 'seq_to_templates.data')
        self.molecules_file = os.path.join(root, 'templates_cano_train.json')
        if load_mol:
            molecules = json.load(open(self.molecules_file))
            self.templates_train = molecules['templates_train']
            self.react_smarts_list = molecules['react_smarts_list']
            self.prod_smarts_list = molecules['prod_smarts_list']
            self.prod_smarts_fp_list = molecules['prod_smarts_fp_list']
            self.fp_prod_smarts_dict = molecules['fp_prod_smarts_dict']
            self.prod_smarts_fp_to_templates = molecules['prod_smarts_fp_to_templates']
            if os.path.exists(self.seq_to_templates_file):
                self.seq_to_templates = torch.load(self.seq_to_templates_file)
            else:
                logging.info(f'can not find seq_to_templates file: {self.seq_to_templates_file}')
                if self.split == 'train':
                    skipped = 0
                    self.seq_to_templates = {}
                    templates_file = os.path.join(self.root, 'templates_train_new.json')
                    with open(templates_file, "r") as f:
                        templates = json.load(f)

                    for idx, val in tqdm(templates.items()):
                        if len(val['templates']) == 0:
                            skipped += 1
                            continue
                        for seq, tmpl in zip(val['template_sequences'], val['templates']):
                            seq = tuple(seq)
                            if seq not in self.seq_to_templates:
                                # for the same seq, there may be multiple templates
                                self.seq_to_templates[seq] = []
                            self.seq_to_templates[seq].append(tmpl)
                    for seq, tmpls in self.seq_to_templates.items():
                        self.seq_to_templates[seq] = sorted(set(tmpls))
                    torch.save(self.seq_to_templates, self.seq_to_templates_file)
                    logging.info(f'unique seq_to_templates size: {len(self.seq_to_templates)}')
                    logging.info(f'skipped train rxns: {skipped}')

        if os.path.isdir(self.processed_dir):
            files = [f for f in os.listdir(self.processed_dir) if f.startswith(split)]
            files = sorted(files)
            self.processed_data_files = [os.path.join(self.processed_dir, f) for f in files]
            if split in ['val', 'test']:
                self.processed_data_files_valid = []
                for data_files in self.processed_data_files:
                    gnn_data = torch.load(data_files)
                    if len(gnn_data.sequences):
                        self.processed_data_files_valid.append(data_files)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        gnn_data = torch.load(self.processed_file_names[idx])
        return gnn_data

    @property
    def processed_file_names(self):
        return self.processed_data_files

    def process_data(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        templates_file = os.path.join(self.root, f'templates_{self.split}_new.json')
        logging.info(f'process datafile: {templates_file}')
        with open(templates_file, "r") as f:
            templates = json.load(f)
        self.processed_data_files = []
        for idx, val in tqdm(templates.items()):
            if self.split == 'train' and len(val['templates']) == 0:
                continue
            p_mol = Chem.MolFromSmiles(val['product'])
            # extract graph features for gnn model
            gnn_data = mol_to_graph_data_obj(p_mol)
            gnn_data.index = idx
            gnn_data.type = val['class']
            gnn_data.product = val['product']
            gnn_data.reactant = val['reactant']
            gnn_data.cano_reactants = val['cano_reactants']
            gnn_data.reaction_smarts = val['reaction_smarts']
            gnn_data.sequences = val['template_sequences']
            gnn_data.templates = val['templates']
            gnn_data.template_cands = val['template_cands']
            gnn_data.reaction_center_cands = val['reaction_center_cands']
            gnn_data.reaction_center_cands_labels = val['reaction_center_cands_labels']
            gnn_data.reaction_center_cands_smarts = val['reaction_center_cands_smarts']
            reaction_center_atom_indexes = torch.zeros(
                (len(val['reaction_center_atom_indexes']), gnn_data.atom_len), dtype=torch.bool)
            for row, atom_indexes in enumerate(val['reaction_center_atom_indexes']):
                reaction_center_atom_indexes[row][atom_indexes] = 1
            gnn_data.reaction_center_atom_indexes = reaction_center_atom_indexes.numpy()
            processed_data_file = os.path.join(self.processed_dir, f'{self.split}_{idx}.data')
            self.processed_data_files.append(processed_data_file)
            torch.save(gnn_data, processed_data_file)

    def decode_reactant_from_seq(self, product, seq, prod_smarts_list, keep_mapnums=False, tmpl_gt=None):
        assert hasattr(self, 'fp_prod_smarts_dict')
        assert hasattr(self, 'react_smarts_list')
        seq[1:] = sorted(seq[1:])
        if tuple(seq) in self.seq_to_templates:
            tmpls = self.seq_to_templates[tuple(seq)]
        else:
            tmpls = []
            reacts_smarts = [str(self.react_smarts_list[s]) for s in seq[1:]]
            reacts_smarts = '.'.join(reacts_smarts)
            for prod_smarts in prod_smarts_list:
                prod_smarts = self.prod_smarts_list[prod_smarts]
                # compose template according to cano product and reactants sub-graphs
                tmpl = compose_tmpl(prod_smarts, reacts_smarts)
                if tmpl:
                    p, r = tmpl.split('>>')
                    mp = Chem.MolFromSmarts(p)
                    mapnums = set(atom.GetAtomMapNum() for atom in mp.GetAtoms() if atom.GetAtomMapNum() > 0)
                    mr = Chem.MolFromSmarts(r)
                    mapnums_r = set(atom.GetAtomMapNum() for atom in mr.GetAtoms() if atom.GetAtomMapNum() > 0)
                    if len(mapnums) == len(mp.GetAtoms()) and mapnums.issubset(mapnums_r):
                        tmpls.append(tmpl)

        results = []
        for tmpl in tmpls:
            # with mapped template, try to run reaction to obtain reactants
            pred_mols = Reactor.run_reaction(product, tmpl, keep_mapnums=keep_mapnums)
            if pred_mols and len(pred_mols):
                results.append((tmpl, pred_mols))

        return results


def prepare_ranking_data(task):
    split, idx, val, processed_dir, typed = task
    # extract graph features for gnn model
    product_mol = Chem.MolFromSmiles(val['product'])
    product_gnn_data = mol_to_graph_data_obj(product_mol)
    product_gnn_data.index = idx
    product_gnn_data.smiles = val['product']
    product_gnn_data.type = val['type']
    product_gnn_data.reactant_gt = val['cano_reactants']
    product_map_numbers = [atom.GetAtomMapNum() for atom in product_mol.GetAtoms()]

    react_mol = Chem.MolFromSmiles(val['reactant'])
    react_map2index = {atom.GetAtomMapNum(): atom.GetIdx() for atom in react_mol.GetAtoms()}
    assert set(product_map_numbers).issubset(react_map2index)
    react_gnn_data = mol_to_graph_data_obj(react_mol)
    react_gnn_data.index = idx
    react_gnn_data.smiles = val['reactant']
    react_gnn_data.order = 1000
    react_gnn_data.log_prob = -1000
    react_gnn_data.product_associated_indexes = []
    for m in product_map_numbers:
        react_gnn_data.product_associated_indexes.append(react_map2index[m])
    # import ipdb; ipdb.set_trace()
    react_gnn_data_list = [react_gnn_data]
    for k, react in enumerate(val['reactants_pred']):
        react_mol = Chem.MolFromSmiles(react)
        react_map2index = {atom.GetAtomMapNum(): atom.GetIdx() for atom in react_mol.GetAtoms()}
        if not set(product_map_numbers).issubset(react_map2index):
            # print('mapping number not contained:', idx, k)
            # import pdb; pdb.set_trace()
            continue

        product_associated_indexes = []
        for m in product_map_numbers:
            product_associated_indexes.append(react_map2index[m])

        react_gnn_data = mol_to_graph_data_obj(react_mol)
        react_gnn_data.index = idx
        react_gnn_data.smiles = react
        react_gnn_data.order = k + 1
        react_gnn_data.log_prob = val['templates_pred_log_prob'][k]
        react_gnn_data.product_associated_indexes = product_associated_indexes
        if val['rank'] == k + 1 and split in ['train', 'valid']:
            assert cano_smiles(react) == val['cano_reactants']
            react_gnn_data_list[0] = react_gnn_data
        else:
            react_gnn_data_list.append(react_gnn_data)

    if split in ['test', 'valid'] or (len(react_gnn_data_list) > 1 and react_gnn_data_list[0].order < 1000):
        filename = 'reaction_{}_{}.data'.format(split, idx)
        if typed:
            filename = 'reaction_typed_{}_{}.data'.format(split, idx)
        processed_data_file = os.path.join(processed_dir, filename)
        torch.save([product_gnn_data, react_gnn_data_list], processed_data_file)


class ReactionDataset(Dataset):
    def __init__(self, root, split, typed=False, topk=50):
        self.split = split
        self.typed = typed
        self.topk = topk
        self.root = os.path.join(root)
        super(ReactionDataset, self).__init__(self.root)
        if os.path.isdir(self.processed_dir):
            filename = 'reaction_{}'.format(split)
            if typed:
                filename = 'reaction_typed_{}'.format(split)
            files = [f for f in os.listdir(self.processed_dir) if f.startswith(filename)]
            files = sorted(files)
            self.processed_data_files = [os.path.join(self.processed_dir, f) for f in files]

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        product, react_gnn_data_list = torch.load(self.processed_file_names[idx])
        product.index = str(product.index)
        if self.split in ['train', 'valid']:
            # use top training predictions, or can randomly sample k predictions
            react_gnn_data_list = react_gnn_data_list[:self.topk]
        else:
            react_gnn_data_list = react_gnn_data_list[:50]
        reactants = Batch.from_data_list(react_gnn_data_list)
        reactants.num_reacts = len(reactants.log_prob)
        reactants.batch_bk = reactants.batch
        del reactants.batch
        reactants.ptr_bk = reactants.ptr
        del reactants.ptr
        return product, reactants

    @property
    def processed_file_names(self):
        return self.processed_data_files

    def process_data(self, beam_results):
        os.makedirs(self.processed_dir, exist_ok=True)
        print('process datafile:', beam_results)
        self.split = beam_results[:-5].split('_')[-1]
        beam_results = json.load(open(beam_results))

        tasks = []
        for idx, val in beam_results.items():
            tasks.append([self.split, idx, val, self.processed_dir, self.typed])

        num_process = 16
        pool = multiprocessing.Pool(processes=num_process)
        pool.map_async(prepare_ranking_data, tasks, len(tasks) // num_process + 1)
        pool.close()
        pool.join()

        filename = 'reaction_{}'.format(self.split)
        if self.typed:
            filename = 'reaction_typed_{}'.format(self.split)
        self.processed_data_files = [f for f in os.listdir(self.processed_dir) if f.startswith(filename)]



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--retro', action='store_true', help='prepare retro data or ranking data')
    parser.add_argument('--typed', action='store_true', help='with reaction types')
    args = parser.parse_args()
    print(args)

    if args.retro:
        print('prepare retrosynthesis data')
        for split in ['train', 'test', 'valid']:
            dataset_valid = MoleculeDataset('data/USPTO50K', split, load_mol=True)
            dataset_valid.process_data()
    else:
        print('prepare ranking data')
        if args.typed:
            # typed case
            dataset_train = ReactionDataset('data/USPTO50K', split='valid', typed=True, topk=50)
            dataset_train.process_data('logs/USPTO50K/uspto50k_typed/beam_result_valid.json')
            dataset_train.process_data('logs/USPTO50K/uspto50k_typed/beam_result_test.json')
            dataset_train.process_data('logs/USPTO50K/uspto50k_typed/beam_result_train.json')
        else:
            # untyped case
            dataset_train = ReactionDataset('data/USPTO50K', split='test', typed=False, topk=50)
            dataset_train.process_data('logs/USPTO50K/uspto50k/beam_result_test.json')
            dataset_train.process_data('logs/USPTO50K/uspto50k/beam_result_valid.json')
            dataset_train.process_data('logs/USPTO50K/uspto50k/beam_result_train.json')
