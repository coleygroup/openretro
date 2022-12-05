import glob
import json
import os
import sys
import torch
import zipfile
from rdkit import Chem
from torch_geometric.data import Batch
from typing import Any, Dict, List


class RetroComposerHandler:
    """RetroComposer Handler for torchserve"""

    def __init__(self):
        self._context = None
        self.initialized = False

        self.model = None
        self.device = None

        self.seq_to_templates = None
        self.templates_train = None
        self.react_smarts_list = None
        self.prod_smarts_list = None
        self.prod_smarts_fp_list = None
        self.fp_prod_smarts_dict = None
        self.prod_smarts_fp_to_templates = None

        self.prod_word_size = None
        self.react_word_size = None

        self.smarts_mol_cache = {}

        # TODO: temporary hardcode
        self.infer_config = {
            "num_layer": 6,
            "emb_dim": 300,
            "atom_feat_dim": 45,
            "bond_feat_dim": 12,
            "center_loss_type": "ce",
            "JK": "concat",
            "dropout_ratio": 0.2,
            "graph_pooling": "attention",
            "beam_size": 50
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

        from models.retrocomposer_model.gnn import GNN_graphpred

        # load_mol (that's what they call it)
        seq_to_templates_file = os.path.join(model_dir, 'seq_to_templates.data')
        self.seq_to_templates = torch.load(seq_to_templates_file)

        molecules_file = os.path.join(model_dir, 'templates_cano_train.json')
        with open(molecules_file, "r") as f:
            molecules = json.load(f)

        self.templates_train = molecules['templates_train']
        self.react_smarts_list = molecules['react_smarts_list']
        self.prod_smarts_list = molecules['prod_smarts_list']
        self.prod_smarts_fp_list = molecules['prod_smarts_fp_list']
        self.fp_prod_smarts_dict = molecules['fp_prod_smarts_dict']
        self.prod_smarts_fp_to_templates = molecules['prod_smarts_fp_to_templates']

        self.prod_word_size = len(self.prod_smarts_fp_list)
        self.react_word_size = len(self.react_smarts_list)

        # build_test_model()
        self.model = GNN_graphpred(
            self.infer_config["num_layer"],
            self.infer_config["emb_dim"],
            self.infer_config["atom_feat_dim"],
            self.infer_config["bond_feat_dim"],
            self.infer_config["center_loss_type"],
            0,
            self.prod_word_size,
            self.react_word_size,
            JK=self.infer_config["JK"],
            drop_ratio=self.infer_config["dropout_ratio"],
            graph_pooling=self.infer_config["graph_pooling"]
        )
        del self.model.gnn_diff
        del self.model.scoring
        self.model = self.model.to(self.device)

        print("Logging model summary")
        print(self.model)
        print(f"\nModel #Params: {sum([x.nelement() for x in self.model.parameters()]) / 1000} k")

        model_file = os.path.join(model_dir, "model.pt")
        self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        print(f"Loaded model from {model_file}")
        sys.stdout.flush()
        self.model.eval()

        self.initialized = True

    def preprocess(self, data: List) -> List[Dict[str, Any]]:
        from utils.chem_utils import canonicalize_smiles

        print(data)
        tmpl_res_list = [self.match_template_for_prediction(canonicalize_smiles(smi))
                         for smi in data[0]["body"]["smiles"]]

        return tmpl_res_list

    def match_template_for_prediction(self, product_smi: str) -> Dict[str, Any]:
        from models.retrocomposer_model.chemutils import get_pattern_fingerprint_bitstr

        params = Chem.SmilesParserParams()
        params.removeHs = False
        mol_prod = Chem.MolFromSmiles(product_smi, params)
        prod_fp_vec = int(get_pattern_fingerprint_bitstr(mol_prod), 2)

        atom_indexes_fp_labels = {}
        # multiple templates may be valid for a reaction, find all of them
        for prod_smarts_fp_idx, prod_smarts_tmpls in self.prod_smarts_fp_to_templates.items():
            prod_smarts_fp_idx = int(prod_smarts_fp_idx)
            prod_smarts_fp = self.prod_smarts_fp_list[prod_smarts_fp_idx]
            for prod_smarts_idx, tmpls in prod_smarts_tmpls.items():
                # skip if fingerprint not match
                if (prod_smarts_fp & prod_fp_vec) < prod_smarts_fp:
                    continue
                prod_smarts_idx = int(prod_smarts_idx)
                prod_smarts = self.prod_smarts_list[prod_smarts_idx]
                if prod_smarts not in self.smarts_mol_cache:
                    self.smarts_mol_cache[prod_smarts] = Chem.MergeQueryHs(Chem.MolFromSmarts(prod_smarts))
                # we need also find matched atom indexes
                matches = mol_prod.GetSubstructMatches(self.smarts_mol_cache[prod_smarts])
                if len(matches):
                    # found_okay_tmpl = False
                    # for tmpl in tmpls:
                    #     pred_mols = Reactor.run_reaction(val['product'], tmpl)
                    #     if reactant and pred_mols and (reactant in pred_mols):
                    #         found_okay_tmpl = True
                    #         template_cands.append(templates_train.index(tmpl))
                    #         templates_list.append(tmpl)
                    #         reacts = tmpl.split('>>')[1].split('.')
                    #         if len(reacts) > 2:
                    #             logging.info(f'too many reacts: {reacts}, {idx}')
                    #         seq_reacts = [react_smarts_list.index(cano_smarts(r)) for r in reacts]
                    #         seq = [prod_smarts_fp_idx] + sorted(seq_reacts)
                    #         sequences.append(seq)
                    # for each prod center, there may be multiple matches
                    for match in matches:
                        match = tuple(sorted(match))
                        if match not in atom_indexes_fp_labels:
                            atom_indexes_fp_labels[match] = {}
                        if prod_smarts_fp_idx not in atom_indexes_fp_labels[match]:
                            atom_indexes_fp_labels[match][prod_smarts_fp_idx] = [[], []]
                        atom_indexes_fp_labels[match][prod_smarts_fp_idx][0].append(prod_smarts_idx)
                        atom_indexes_fp_labels[match][prod_smarts_fp_idx][1].append(False)

        reaction_center_cands = []
        reaction_center_cands_smarts = []
        reaction_center_atom_indexes = []
        for atom_index in sorted(atom_indexes_fp_labels.keys()):
            for fp_idx, val in atom_indexes_fp_labels[atom_index].items():
                reaction_center_cands.append(fp_idx)
                reaction_center_cands_smarts.append(val[0])
                reaction_center_atom_indexes.append(atom_index)

        tmpl_res = {
            "product": product_smi,
            'reaction_center_cands': reaction_center_cands,
            'reaction_center_cands_smarts': reaction_center_cands_smarts,
            'reaction_center_atom_indexes': reaction_center_atom_indexes
        }

        return tmpl_res

    def inference(self, data: List[Dict[str, Any]]):
        from models.retrocomposer_model.chemutils import cano_smiles
        from models.retrocomposer_model.prepare_mol_graph import mol_to_graph_data_obj
        from utils.chem_utils import canonicalize_smiles

        gnn_data_batch = []
        for idx, tmpl_res in enumerate(data):
            # print(tmpl_res)
            p_mol = Chem.MolFromSmiles(tmpl_res["product"])
            gnn_data = mol_to_graph_data_obj(p_mol)
            gnn_data.index = idx
            gnn_data.product = tmpl_res['product']
            gnn_data.reaction_center_cands = tmpl_res['reaction_center_cands']
            gnn_data.reaction_center_cands_smarts = tmpl_res['reaction_center_cands_smarts']
            # print(gnn_data)
            reaction_center_atom_indexes = torch.zeros(
                (len(tmpl_res['reaction_center_atom_indexes']), gnn_data.atom_len), dtype=torch.bool)
            for row, atom_indexes in enumerate(tmpl_res['reaction_center_atom_indexes']):
                reaction_center_atom_indexes[row][list(atom_indexes)] = 1
            gnn_data.reaction_center_atom_indexes = reaction_center_atom_indexes.numpy()
            gnn_data_batch.append(gnn_data)

        # eval_decoding
        batch = Batch.from_data_list(gnn_data_batch)
        batch = batch.to(self.device)

        pred_results = {}
        with torch.no_grad():
            beam_nodes = self.model(batch, typed=False, decode=True, beam_size=self.infer_config["beam_size"])
            cano_pred_mols = {}
            for node in beam_nodes:
                batch_idx = node.index
                data_idx = batch.index[batch_idx].item()
                if data_idx not in cano_pred_mols:
                    cano_pred_mols[data_idx] = set()
                if data_idx not in pred_results:
                    pred_results[data_idx] = {
                        'rank': 1000,
                        'product': batch.product[batch_idx],
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
                seq_pred[1:] = [tp for tp in seq_pred[1:] if tp < len(self.react_smarts_list)]
                decoded_results = self._decode_reactant_from_seq(
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
                            # if pred_results[data_idx]['cano_reactants'] == cano_pred_mol:
                            #     pred_results[data_idx]['rank'] = min(
                            #         pred_results[data_idx]['rank'], len(pred_results[data_idx]['seq_pred']))

            beam_nodes.clear()

        results = []
        for data_idx, pred_result in pred_results.items():
            result = {
                "templates": pred_result['templates_pred'],
                "reactants": [canonicalize_smiles(r, remove_atom_number=True)
                              for r in pred_result['reactants_pred']],
                "scores": pred_result['templates_pred_log_prob']
            }
            results.append(result)
        return results

    def _decode_reactant_from_seq(self, product, seq, prod_smarts_list, keep_mapnums=False, tmpl_gt=None):
        from models.retrocomposer_model.extract_templates import Reactor
        from models.retrocomposer_model.process_templates import compose_tmpl

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

    def postprocess(self, data: List):
        return [data]

    def handle(self, data, context) -> List[List[Dict[str, Any]]]:
        self._context = context

        output = self.preprocess(data)
        output = self.inference(output)
        output = self.postprocess(output)

        print(output)
        return output
