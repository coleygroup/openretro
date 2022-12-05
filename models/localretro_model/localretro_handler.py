import dgl
import glob
import os
import pandas as pd
import sys
import torch
import torch.nn as nn
import zipfile
from dgllife.utils import WeaveAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph
from typing import Any, Dict, List


class LocalRetroHandler:
    """LocalRetro Handler for torchserve"""

    def __init__(self):
        self._context = None
        self.initialized = False

        self.model = None
        self.device = None
        self.node_featurizer = None
        self.edge_featurizer = None
        self.atom_templates = None
        self.bond_templates = None
        self.template_infos = None

        # TODO: temporary hardcode
        self.infer_config = {
            "node_out_feats": 320,
            "edge_hidden_feats": 64,
            "num_step_message_passing": 6,
            "attention_heads": 8,
            "attention_layers": 1,
            "top_num": 100,
            "top_k": 50
        }

    def initialize(self, context):
        self._context = context
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        print(glob.glob(f"{model_dir}/*"))
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        with zipfile.ZipFile(model_dir + '/models.zip', 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        with zipfile.ZipFile(model_dir + '/utils.zip', 'r') as zip_ref:
            zip_ref.extractall(model_dir)

        from models.localretro_model.model import LocalRetro

        atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
                      'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
                      'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                      'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba',
                      'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs']
        self.node_featurizer = WeaveAtomFeaturizer(atom_types=atom_types)
        self.edge_featurizer = CanonicalBondFeaturizer(self_loop=True)

        atom_templates = pd.read_csv(os.path.join(model_dir, "atom_templates.csv"))
        bond_templates = pd.read_csv(os.path.join(model_dir, "bond_templates.csv"))
        template_infos = pd.read_csv(os.path.join(model_dir, "template_infos.csv"))
        AtomTemplate_n = len(atom_templates)
        BondTemplate_n = len(bond_templates)

        self.atom_templates = {atom_templates['Class'][i]: atom_templates['Template'][i]
                               for i in atom_templates.index}
        self.bond_templates = {bond_templates['Class'][i]: bond_templates['Template'][i]
                               for i in bond_templates.index}
        self.template_infos = {template_infos['Template'][i]: {
            'edit_site': eval(template_infos['edit_site'][i]),
            'change_H': eval(template_infos['change_H'][i]),
            'change_C': eval(template_infos['change_C'][i]),
            'change_S': eval(template_infos['change_S'][i])}
            for i in template_infos.index}

        self.model = LocalRetro(
            node_in_feats=self.node_featurizer.feat_size(),
            edge_in_feats=self.edge_featurizer.feat_size(),
            node_out_feats=self.infer_config["node_out_feats"],
            edge_hidden_feats=self.infer_config["edge_hidden_feats"],
            num_step_message_passing=self.infer_config["num_step_message_passing"],
            attention_heads=self.infer_config["attention_heads"],
            attention_layers=self.infer_config["attention_layers"],
            AtomTemplate_n=AtomTemplate_n,
            BondTemplate_n=BondTemplate_n
        )

        print("Logging model summary")
        print(self.model)
        print(f"\nModel #Params: {sum([x.nelement() for x in self.model.parameters()]) / 1000} k")

        checkpoint = os.path.join(model_dir, "LocalRetro.pth")
        state_dict = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state_dict['model_state_dict'])
        print(f"Loaded state_dict from {checkpoint}")
        sys.stdout.flush()

        self.model = self.model.to(self.device)
        self.model.eval()

        self.initialized = True

    def preprocess(self, data: List) -> List[str]:
        from utils.chem_utils import canonicalize_smiles

        print(data)
        canonical_smiles = [canonicalize_smiles(smi)
                            for smi in data[0]["body"]["smiles"]]

        return canonical_smiles

    def inference(self, data: List[str]):
        from models.localretro_model.get_edit import combined_edit, get_bg_partition
        from models.localretro_model.LocalTemplate.template_decoder import read_prediction, decode_localtemplate

        print("Building dgl graphs for canonical smiles...")
        graphs = [smiles_to_bigraph(smi, add_self_loop=True,
                                    node_featurizer=self.node_featurizer,
                                    edge_featurizer=self.edge_featurizer,
                                    canonical_atom_order=False)
                  for smi in data]
        graphs = dgl.batch(graphs)
        graphs.set_n_initializer(dgl.init.zero_initializer)
        graphs.set_e_initializer(dgl.init.zero_initializer)
        graphs = graphs.to(self.device)
        node_feats = graphs.ndata.pop("h").to(self.device)
        edge_feats = graphs.edata.pop("e").to(self.device)

        results = []
        with torch.no_grad():
            batch_atom_logits, batch_bond_logits, _ = self.model(graphs, node_feats, edge_feats)
            batch_atom_logits = nn.Softmax(dim=1)(batch_atom_logits)
            batch_bond_logits = nn.Softmax(dim=1)(batch_bond_logits)
            graphs, nodes_sep, edges_sep = get_bg_partition(graphs)

            start_node = 0
            start_edge = 0

            for smi, graph, end_node, end_edge in zip(data, graphs, nodes_sep, edges_sep):
                # raw predictions
                pred_types, pred_sites, pred_scores = combined_edit(
                    graph,
                    atom_out=batch_atom_logits[start_node:end_node],
                    bond_out=batch_bond_logits[start_edge:end_edge],
                    top_num=self.infer_config["top_num"]
                )
                start_node = end_node
                start_edge = end_edge

                predictions = [(pred_types[i], pred_sites[i][0], pred_sites[i][1], pred_scores[i])
                               for i in range(self.infer_config["top_num"])]

                # template application and reactant recovery
                reactants = []
                scores = []
                for prediction in predictions:
                    mol, pred_site, template, template_info, score = read_prediction(
                        smiles=smi,
                        prediction=prediction,
                        atom_templates=self.atom_templates,
                        bond_templates=self.bond_templates,
                        template_infos=self.template_infos,
                        raw=True
                    )
                    local_template = '>>'.join(['(%s)' % smarts for smarts in template.split('_')[0].split('>>')])
                    try:
                        decoded_smiles = decode_localtemplate(mol, pred_site, local_template, template_info)
                        if decoded_smiles is None or str((decoded_smiles, score)) in reactants:
                            continue
                    except Exception as e:
                        continue
                    reactants.append(decoded_smiles)
                    scores.append(score.item())

                    if len(reactants) >= self.infer_config["top_k"]:
                        break
                result = {"reactants": reactants, "scores": scores}
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
