import glob

import dgl
import logging
import networkx as nx
import numpy as np
import os
import pickle
from collections import Counter
from torch.utils.data import Dataset
from typing import List


def len2idx(lens) -> np.ndarray:
    # end_indices = np.cumsum(np.concatenate(lens, axis=0))
    end_indices = np.cumsum(lens)
    start_indices = np.concatenate([[0], end_indices[:-1]], axis=0)
    indices = np.stack([start_indices, end_indices], axis=1)

    return indices


class RetroCenterDatasets(Dataset):
    def __init__(self, processed_data_path: str, data_split: str):
        self.fp = os.path.join(processed_data_path, data_split)
        logging.info(f"Creating dataset from folder {self.fp}")
        self.n_files = len(glob.glob(os.path.join(self.fp, "*.pkl")))

        # fn_pattern = os.path.join(processed_data_path, f"pattern_feat_{data_split}.npz")
        # logging.info(f"Loading pattern features from {fn_pattern}")
        # feat = np.load(fn_pattern)
        # self.pattern_features = feat["pattern_features"]
        # self.pattern_features_lens = feat["pattern_features_lens"]
        # self.pattern_features_indices = len2idx(self.pattern_features_lens)

        self.disconnection_num = []
        cnt = Counter()
        for i in range(self.n_files):
            fn = os.path.join(self.fp, f"rxn_data_{i}.pkl")
            with open(fn, "rb") as f:
                rxn_data = pickle.load(f)

            if "target_adj" in rxn_data:
                xa = rxn_data["product_adj"]
                ya = rxn_data["target_adj"]
                res = xa & (ya == False)
                res = np.sum(np.sum(res)) // 2
                cnt[res] += 1
                if res >= 2:
                    res = 2
            else:
                res = -1
            self.disconnection_num.append(res)
        logging.info(cnt)

    def __getitem__(self, index):
        fn = os.path.join(self.fp, f"rxn_data_{index}.pkl")
        # with open(self.fl[index], "rb") as f:
        with open(fn, "rb") as f:
            rxn_data = pickle.load(f)
        if "target_adj" not in rxn_data:
            return self.__getitem__(0)

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

        # start, end = self.pattern_features_indices[index]
        # x_pattern_feat = self.pattern_features[start:end]

        return rxn_class, x_pattern_feat, x_atom, x_adj, x_graph, y_adj, disconnection_num

    def __len__(self):
        return self.n_files


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
