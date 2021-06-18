import glob
import numpy as np
import os
import scipy
import torch
import torch.nn as nn
from chem_utils import canonicalize_smiles
from neuralsym_model.model import TemplateNN_Highway
from rdchiral.main import rdchiralReaction, rdchiralReactants, rdchiralRun
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from scipy import sparse
from typing import Any, Dict, List


def mol_smi_to_count_fp(
    mol_smi: str, radius: int = 2, fp_size: int = 32681, dtype: str = "int32"
) -> scipy.sparse.csr_matrix:
    fp_gen = GetMorganGenerator(
        radius=radius, useCountSimulation=True, includeChirality=True, fpSize=fp_size
    )
    mol = Chem.MolFromSmiles(mol_smi)
    uint_count_fp = fp_gen.GetCountFingerprint(mol)
    count_fp = np.empty((1, fp_size), dtype=dtype)
    DataStructs.ConvertToNumpyArray(uint_count_fp, count_fp)
    return sparse.csr_matrix(count_fp, dtype=dtype)


class NeuralSymHandler:
    """NeuralSym Handler for torchserve"""

    def __init__(self):
        self._context = None
        self.initialized = False

        self.templates_filtered = []
        self.model = None
        self.indices = None
        self.device = None

        # TODO: temporary hardcode
        self.use_cpu = True
        self.n_best = 20

        self.infer_config = {
            'data_name': "schneider50k",
            'min_freq': 1,
            'hidden_size': 300,
            'depth': 0,
            'orig_fp_size': 1000000,
            'final_fp_size': 32681,
            'radius': 2,
        }

    def initialize(self, context):
        self._context = context
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        print(glob.glob(f"{model_dir}/*"))
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        template_file = os.path.join(model_dir, "training_templates.txt")
        print(f"Loading templates from {template_file}")
        with open(template_file, 'r') as f:
            templates = f.readlines()

        for p in templates:
            pa, cnt = p.strip().split(': ')
            if int(cnt) >= self.infer_config['min_freq']:
                self.templates_filtered.append(pa)
        print(f'Total number of template patterns: {len(self.templates_filtered)}')

        checkpoint_file = os.path.join(model_dir, f"{self.infer_config['data_name']}.pth.tar")
        print(f"Building model and loading from {checkpoint_file}")
        self.model = TemplateNN_Highway(
            output_size=len(self.templates_filtered),
            size=self.infer_config['hidden_size'],
            num_layers_body=self.infer_config['depth'],
            input_size=self.infer_config['final_fp_size']
        )
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        indices_file = os.path.join(model_dir, "variance_indices.txt")
        print(f"Loading indices from {indices_file}")
        self.indices = np.loadtxt(indices_file).astype("int")

        self.initialized = True

    def preprocess(self, data: List):
        print(data)
        canonical_smiles = [canonicalize_smiles(smi)
                            for smi in data[0]["body"]["smiles"]]

        return canonical_smiles

    def inference(self, data: List[str]):
        results = []

        with torch.no_grad():
            for smi in data:
                prod_fp = mol_smi_to_count_fp(smi, self.infer_config['radius'], self.infer_config['orig_fp_size'])
                logged = sparse.csr_matrix(np.log(prod_fp.toarray() + 1))
                final_fp = logged[:, self.indices]
                final_fp = torch.as_tensor(final_fp.toarray()).float().to(self.device)

                outputs = self.model(final_fp)
                outputs = nn.Softmax(dim=1)(outputs)
                preds = torch.topk(outputs, k=self.n_best, dim=1)[1].squeeze(dim=0).cpu().numpy()

                result = {
                    "reactants": [],
                    "scores": []
                }
                for idx in preds:
                    score = outputs[0, idx.item()].item()
                    template = self.templates_filtered[idx.item()]
                    rxn = rdchiralReaction(template)
                    prod = rdchiralReactants(smi)
                    try:
                        precs = rdchiralRun(rxn, prod)
                    except:
                        precs = 'N/A'

                    if not precs:           # empty precursors?
                        continue
                    result["reactants"].extend(precs)
                    result["scores"].append(score)

                results.append(result)

        return results

    def postprocess(self, data: List):
        return [data]

    def handle(self, data, context) -> List[Dict[str, Any]]:
        self._context = context

        output = self.preprocess(data)
        output = self.inference(output)
        output = self.postprocess(output)

        print(output)
        return output
