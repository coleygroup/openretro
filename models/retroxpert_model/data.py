import dgl
import logging
import networkx as nx
import numpy as np
import os
import pickle
from collections import Counter
from torch.utils.data import Dataset


class RetroCenterDatasets(Dataset):
    """Modified dataset, now there is a single .pkl file per phase"""
    def __init__(self, processed_data_path: str, fn: str):
        fn = os.path.join(processed_data_path, fn)
        logging.info(f"Creating dataset from {fn}")
        with open(fn, "rb") as f:
            self.rxn_data_dict = pickle.load(f)

        self.disconnection_num = []
        cnt = Counter()
        for i, rxn_data in self.rxn_data_dict.items():
            xa = rxn_data["product_adj"]
            ya = rxn_data["target_adj"]
            res = xa & (ya == False)
            res = np.sum(np.sum(res)) // 2
            cnt[res] += 1
            if res >= 2:
                res = 2
            self.disconnection_num.append(res)
        logging.info(cnt)

    def __getitem__(self, index):
        rxn_data = self.rxn_data_dict[index]

        x_atom = rxn_data["product_atom_features"].astype(np.float32)
        x_pattern_feat = rxn_data["pattern_feat"].astype(np.float32)
        x_bond = rxn_data["product_bond_features"].astype(np.float32)
        x_adj = rxn_data["product_adj"]
        y_adj = rxn_data["target_adj"]
        rxn_class = rxn_data["rxn_type"]
        if rxn_class == "UNK":
            rxn_class = 0
        rxn_class = np.eye(10)[rxn_class]

        product_atom_num = len(x_atom)
        rxn_class = np.expand_dims(rxn_class, 0).repeat(product_atom_num, axis=0)
        disconnection_num = self.disconnection_num[index]
        # Construct graph and add edge data
        x_graph = dgl.DGLGraph(nx.from_numpy_matrix(x_adj))
        x_graph.edata["w"] = x_bond[x_adj]
        return rxn_class, x_pattern_feat, x_atom, x_adj, x_graph, y_adj, disconnection_num

    def __len__(self):
        return len(self.rxn_data_dict)


class RetroCenterDatasetsOriginal(Dataset):
    def __init__(self, root, data_split):
        self.root = root
        self.data_split = data_split

        self.data_dir = os.path.join(root, self.data_split)
        self.data_files = [
            f for f in os.listdir(self.data_dir) if f.endswith('.pkl')
        ]
        self.data_files.sort()

        self.disconnection_num = []
        cnt = Counter()
        for data_file in self.data_files:
            with open(os.path.join(self.data_dir, data_file), 'rb') as f:
                reaction_data = pickle.load(f)
            xa = reaction_data['product_adj']
            ya = reaction_data['target_adj']
            res = xa & (ya == False)
            res = np.sum(np.sum(res)) // 2
            cnt[res] += 1
            if res >= 2:
                res = 2
            self.disconnection_num.append(res)
        print(cnt)

    def __getitem__(self, index):
        with open(os.path.join(self.data_dir, self.data_files[index]),
                  'rb') as f:
            reaction_data = pickle.load(f)

        x_atom = reaction_data['product_atom_features'].astype(np.float32)
        x_pattern_feat = reaction_data['pattern_feat'].astype(np.float32)
        x_bond = reaction_data['product_bond_features'].astype(np.float32)
        x_adj = reaction_data['product_adj']
        y_adj = reaction_data['target_adj']
        rxn_class = reaction_data['rxn_type']
        rxn_class = np.eye(10)[rxn_class]
        product_atom_num = len(x_atom)
        rxn_class = np.expand_dims(rxn_class, 0).repeat(product_atom_num,
                                                        axis=0)
        disconnection_num = self.disconnection_num[index]
        # Construct graph and add edge data
        x_graph = dgl.DGLGraph(nx.from_numpy_matrix(x_adj))
        x_graph.edata['w'] = x_bond[x_adj]
        return rxn_class, x_pattern_feat, x_atom, x_adj, x_graph, y_adj, disconnection_num

    def __len__(self):
        return len(self.data_files)


if __name__ == '__main__':
    savedir = 'data/USPTO50K/'
    for data_set in ['train', 'test', 'valid']:
        save_dir = os.path.join(savedir, data_set)
        train_data = RetroCenterDatasets(root=savedir, data_split=data_set)
        print(train_data.data_files[:100])
