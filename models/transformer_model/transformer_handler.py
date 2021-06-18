import glob
import os
import torch
from chem_utils import canonicalize_smiles, smi_tokenizer
from onmt.translate.translation_server import ServerModel as ONMTServerModel
from typing import Any, Dict, List


class TransformerHandler:
    """Transformer Handler for torchserve"""

    def __init__(self):
        self._context = None
        self.initialized = False

        self.model = None
        self.device = None

        # TODO: temporary hardcode
        self.use_cpu = True
        self.n_best = 20
        self.beam_size = 5

    def initialize(self, context):
        self._context = context
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        print(glob.glob(f"{model_dir}/*"))
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        onmt_config = {
            "models": os.path.join(model_dir, "model_step_125000.pt"),
            "n_best": self.n_best,
            "beam_size": self.beam_size,
            "gpu": -1 if self.use_cpu else 0
        }

        self.model = ONMTServerModel(
            opt=onmt_config,
            model_id=0,
            load=True
        )

        self.initialized = True

    def preprocess(self, data: List):
        print(data)
        tokenized_smiles = [{"src": smi_tokenizer(canonicalize_smiles(smi))}
                            for smi in data[0]["body"]["smiles"]]

        return tokenized_smiles

    def inference(self, data: List[Dict[str, str]]):

        reactants, scores, _, _, _ = self.model.run(inputs=data)

        results = []
        for i in range(len(data)):  # essentially reshaping (b*n_best,) into (b, n_best)
            start = self.n_best * i
            end = self.n_best * (i + 1)
            result = {
                "reactants": ["".join(r.split()) for r in reactants[start:end]],
                "scores": scores[start:end]
            }
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
