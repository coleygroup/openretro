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
        self.n_best = 10
        self.beam_size = 20

    def initialize(self, context):
        self._context = context
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        print(glob.glob(f"{model_dir}/*"))
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        checkpoint_file = os.path.join(model_dir, "model_step_125000.pt")
        if not os.path.isfile(checkpoint_file):
            checkpoint_list = sorted(glob.glob(os.path.join(model_dir, f"model_step_*.pt")))
            print(f"Default checkpoint file {checkpoint_file} not found!")
            print(f"Using found last checkpoint {checkpoint_list[-1]} instead.")
            checkpoint_file = checkpoint_list[-1]

        onmt_config = {
            "models": checkpoint_file,
            "n_best": self.n_best * 2,
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
