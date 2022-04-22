import dgl
import glob
import networkx as nx
import numpy as np
import os
import torch
from chem_utils import canonicalize_smiles
from onmt.translate.translation_server import ServerModel as ONMTServerModel
from retroxpert_model.model.gat import GATNet
from retroxpert_model.preprocessing import get_atom_features, get_bond_features, get_smarts_pieces_s2, \
    smi_tokenizer, smarts2smiles
from rdkit import Chem
from typing import Any, Dict, List


class RetroXpertHandler:
    """RetroXpert Handler for torchserve"""

    def __init__(self):
        self._context = None
        self.initialized = False

        self.patterns_filtered = []
        self.model_stage_1 = None
        self.model_stage_2 = None
        self.device = None

        # TODO: temporary hardcode
        self.in_dim = 47
        self.gat_layers = 3
        self.heads = 4
        self.hidden_dim = 128
        self.use_cpu = True
        self.n_best = 20
        # self.beam_size = 50
        self.beam_size = 10
        self.min_freq = 10

    def initialize(self, context):
        self._context = context
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        print(glob.glob(f"{model_dir}/*"))
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        print("Loading pattern")
        pattern_file = os.path.join(model_dir, "product_patterns.txt")
        with open(pattern_file, "r") as f:
            for line in f:
                pattern, count = line.strip().split(": ")
                if int(count.strip()) >= self.min_freq:
                    self.patterns_filtered.append(pattern)
        print(f"Filtered patterns by min frequency {self.min_freq}, "
              f"remaining pattern count: {len(self.patterns_filtered)}")

        self.model_stage_1 = GATNet(
            in_dim=self.in_dim + len(self.patterns_filtered),
            num_layers=self.gat_layers,
            hidden_dim=self.hidden_dim,
            heads=self.heads,
            use_gpu=(not self.use_cpu),
        )
        self.model_stage_1 = self.model_stage_1.to(self.device)
        self.model_stage_1.eval()

        print("Logging model summary")
        print(self.model_stage_1)
        print(f"\nModel #Params: {sum([x.nelement() for x in self.model_stage_1.parameters()]) / 1000} k")

        data_name = context.model_name.replace('_retroxpert', '')
        checkpoint_path = os.path.join(model_dir, f"{data_name}_untyped_checkpoint.pt")
        print(f"Loading from {checkpoint_path}")
        self.model_stage_1.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        onmt_config = {
            "models": os.path.join(model_dir, "model_step_300000.pt"),
            "n_best": self.n_best * 2,
            "beam_size": self.beam_size
        }

        self.model_stage_2 = ONMTServerModel(
            opt=onmt_config,
            model_id=0,
            load=True
        )

        self.initialized = True

    def find_patterns(self, product_smi: str, product_mol):
        [a.SetAtomMapNum(0) for a in product_mol.GetAtoms()]
        matches_all = {}
        for idx, pattern in enumerate(self.patterns_filtered):
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol is None:
                print(f"error: pattern_mol is None, idx: {idx}")
            try:
                matches = product_mol.GetSubstructMatches(pattern_mol,
                                                          useChirality=False)
            except:
                print(f"Caught some exception for {product_smi}. Continue anyway.")
                continue
            else:
                if len(matches) > 0 and len(matches[0]) > 0:
                    matches_all[idx] = matches
        if len(matches_all) == 0:
            print(product_smi)

        num_atoms = product_mol.GetNumAtoms()
        pattern_feature = np.zeros((len(self.patterns_filtered), num_atoms))
        for idx, matches in matches_all.items():
            if len(matches) > 1 and isinstance(matches[0], tuple):
                for match in matches:
                    np.put(pattern_feature[idx], match, 1)
            else:
                np.put(pattern_feature[idx], matches, 1)
        pattern_feature = pattern_feature.transpose().astype('bool_')

        return pattern_feature

    def preprocess(self, data: List):
        print(data)
        rxn_data_dict = {}

        for i, smi in enumerate(data[0]["body"]["smiles"]):
            reaction_class = "UNK"

            # canonicalization
            mol = Chem.MolFromSmiles(smi)
            index2mapnums = {}
            for atom in mol.GetAtoms():
                index2mapnums[atom.GetIdx()] = atom.GetAtomMapNum()

            mol_cano = Chem.RWMol(mol)
            [atom.SetAtomMapNum(0) for atom in mol_cano.GetAtoms()]
            smi_cano = Chem.MolToSmiles(mol_cano)
            mol_cano = Chem.MolFromSmiles(smi_cano)

            matches = mol.GetSubstructMatches(mol_cano)
            if matches:
                for atom, mat in zip(mol_cano.GetAtoms(), matches[0]):
                    atom.SetAtomMapNum(index2mapnums[mat])
                smi = Chem.MolToSmiles(mol_cano, canonical=False)
            # /canonicalization

            product_mol = Chem.MolFromSmiles(smi)
            product_adj = Chem.rdmolops.GetAdjacencyMatrix(product_mol)
            product_adj = product_adj + np.eye(product_adj.shape[0])
            product_adj = product_adj.astype(np.bool)

            product_bond_features = get_bond_features(product_mol)
            product_atom_features = get_atom_features(product_mol)

            rxn_data = {
                "product_smi": smarts2smiles(smi),
                'rxn_type': reaction_class,
                'product_adj': product_adj,
                'product_mol': product_mol,
                'product_bond_features': product_bond_features,
                'product_atom_features': product_atom_features,
                "pattern_feat": self.find_patterns(smi, product_mol)
            }
            rxn_data_dict[i] = rxn_data

        return rxn_data_dict

    def inference(self, data: Dict[int, Dict[str, Any]]):
        # Stage 1
        x_pattern_feats = []
        x_atoms = []
        x_adjs = []
        x_graphs = []

        product_smis = []
        product_adjs = []
        product_mols = []
        pred_logits_mol_list = []

        # Collation, adapted from RetroCenterDatasets.__getitem__()
        for i, rxn_data in data.items():
            x_atom = rxn_data["product_atom_features"].astype(np.float32)
            x_bond = rxn_data["product_bond_features"].astype(np.float32)
            x_pattern_feat = rxn_data["pattern_feat"].astype(np.float32)
            x_adj = rxn_data["product_adj"]

            # Construct graph and add edge data
            x_graph = dgl.DGLGraph(nx.from_numpy_matrix(x_adj))
            x_graph.edata["w"] = x_bond[x_adj]

            x_pattern_feats.append(x_pattern_feat)
            x_atoms.append(x_atom)
            x_adjs.append(x_adj)
            x_graphs.append(x_graph)

            product_smis.append(rxn_data["product_smi"])
            product_adjs.append(rxn_data["product_adj"])
            product_mols.append(rxn_data["product_mol"])

        # aliasing to be consistent with original
        x_pattern_feat = x_pattern_feats
        x_atom = x_atoms
        x_adj = x_adjs
        x_graph = x_graphs

        # adapted from testing loop
        x_atom = list(map(lambda x: torch.from_numpy(x).float(), x_atom))
        x_pattern_feat = list(map(lambda x: torch.from_numpy(x).float(), x_pattern_feat))
        x_atom = list(map(lambda x, y: torch.cat([x, y], dim=1), x_atom, x_pattern_feat))
        x_atom = torch.cat(x_atom, dim=0)
        x_atom = x_atom.to(self.device)

        x_adj = list(map(lambda x: torch.from_numpy(np.array(x)), x_adj))
        x_adj = [xa.to(self.device) for xa in x_adj]

        # batch graph
        g_dgl = dgl.batch(x_graph)
        h_pred, e_pred = self.model_stage_1(g_dgl, x_atom)

        e_pred = e_pred.squeeze()
        h_pred = torch.argmax(h_pred, dim=1)
        # bond_change_pred_list.extend(h_pred.cpu().tolist())
        bond_change_pred_list = h_pred.cpu().tolist()

        end = 0
        pred = torch.sigmoid(e_pred)

        for adj in x_adj:
            mask_pos = torch.nonzero(adj).tolist()

            start = end
            end = start + len(mask_pos)
            pred_proab = pred[start:end]

            assert len(mask_pos) == len(pred_proab)

            pred_disconnection_adj = torch.zeros_like(adj, dtype=torch.float32)
            for idx, pos in enumerate(mask_pos):
                pred_disconnection_adj[pos[0], pos[1]] = pred_proab[idx]
            for idx, pos in enumerate(mask_pos):
                pred_proab[idx] = (pred_disconnection_adj[pos[0], pos[1]] +
                                   pred_disconnection_adj[pos[1], pos[0]]) / 2

            pred_logits_mol_list.append(pred_proab.tolist())

        bond_disconnection = []
        for bond_change_num, pred_adj_list in zip(bond_change_pred_list, pred_logits_mol_list):
            pred_adj_index = np.argsort(pred_adj_list)
            pred_adj_index = pred_adj_index[:bond_change_num]

            bond_disconnection.append(pred_adj_index)

        print("Generate synthons from bond disconnection prediction")
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

        # Stage 2
        input_smiles = []
        for smi, synthon in zip(product_smis, synthons):
            input_smiles.append(f"[RXN_0] {smi_tokenizer(smi)} [PREDICT] {smi_tokenizer(synthon)}")

        print(input_smiles)

        inputs = [{"src": smi} for smi in input_smiles]

        reactants, scores, _, _, _ = self.model_stage_2.run(inputs=inputs)

        results = []
        for i, prod in enumerate(input_smiles):  # essentially reshaping (b*n_best,) into (b, n_best)
            valid_reactants = []
            valid_scores = []

            for j in range(self.n_best * 2 * i, self.n_best * 2 * (i + 1)):
                if len(valid_reactants) == self.n_best:
                    break

                reactant = "".join(reactants[j].split())
                if not canonicalize_smiles(reactant):
                    continue
                else:
                    valid_reactants.append(reactant)
                    valid_scores.append(scores[j])

            result = {
                "reactants": valid_reactants,
                "scores": valid_scores
            }
            results.append(result)

        return results

    def postprocess(self, data: List):
        return [data]

    def handle(self, data, context) -> List[List[Dict[str, Any]]]:
        self._context = context

        output = self.preprocess(data)
        output = self.inference(output)
        output = self.postprocess(output)

        print(output)
        return output
