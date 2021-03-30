import glob
import torch
from gln.test.model_inference import RetroGLN
from typing import Any, Dict, List


class GLNHandler:
    """GLN Handler for torchserve"""

    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None

    def initialize(self, context):
        self._context = context
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        print(glob.glob(f"{model_dir}/*"))
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        self.dropbox = model_dir

        self.model = RetroGLN(dropbox=self.dropbox,
                              model_dump=model_dir)

        self.initialized = True

    def preprocess(self, data: List):
        return data

    def inference(self, data: List[str]):
        print(data)
        topk = 10
        beam_size = 10
        rxn_type = "UNK"

        results = []
        for smi in data[0]["body"]["smiles"]:
            result = self.model.run(raw_prod=smi,
                                    beam_size=beam_size,
                                    topk=topk,
                                    rxn_type=rxn_type)
            result["scores"] = result["scores"].tolist()
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
