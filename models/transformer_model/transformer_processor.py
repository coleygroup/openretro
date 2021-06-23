import csv
import logging
import os
import re
from base.processor_base import Processor
from onmt.bin.preprocess import preprocess as onmt_preprocess
from rdkit import Chem
from tqdm import tqdm
from typing import Dict, List


def smi_tokenizer(smi: str):
    """Tokenize a SMILES molecule or reaction, adapted from https://github.com/pschwllr/MolecularTransformer"""
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == "".join(tokens)

    return " ".join(tokens)


class TransformerProcessor(Processor):
    """Class for Transformer Preprocessing"""

    def __init__(self,
                 model_name: str,
                 model_args,
                 model_config: Dict[str, any],
                 data_name: str,
                 raw_data_files: List[str],
                 processed_data_path: str,
                 num_cores: int = None):
        super().__init__(model_name=model_name,
                         model_args=model_args,
                         model_config=model_config,
                         data_name=data_name,
                         raw_data_files=raw_data_files,
                         processed_data_path=processed_data_path)
        self.check_count = 100
        self.train_file, self.val_file, self.test_file = raw_data_files
        self.num_cores = num_cores

        logging.info("Overwriting model args, (hardcoding essentially)")
        self.overwrite_model_args()
        logging.info(f"Updated model args: {self.model_args}")

    def overwrite_model_args(self):
        """Overwrite model args"""
        # Paths
        self.model_args.save_data = os.path.join(self.processed_data_path, "bin")
        self.model_args.train_src = [os.path.join(self.processed_data_path, f"src-train.txt")]
        self.model_args.train_tgt = [os.path.join(self.processed_data_path, f"tgt-train.txt")]
        self.model_args.valid_src = os.path.join(self.processed_data_path, f"src-val.txt")
        self.model_args.valid_tgt = os.path.join(self.processed_data_path, f"tgt-val.txt")
        # Runtime args
        self.model_args.overwrite = True
        self.model_args.share_vocab = True
        self.model_args.subword_prefix = "ThisIsAHardCode"          # an arg for BART, leading to weird logging error

    def check_data_format(self) -> None:
        """Check that all files exists and the data format is correct for all"""
        super().check_data_format()

        logging.info(f"Checking the first {self.check_count} entries for each file")
        for fn in self.raw_data_files:
            if not fn:
                continue

            with open(fn, "r") as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for i, row in enumerate(csv_reader):
                    if i > self.check_count:            # check the first few rows
                        break

                    assert (c in row for c in ["class", "rxn_smiles"]), \
                        f"Error processing file {fn} line {i}, ensure columns 'class' and " \
                        f"'rxn_smiles' is included!"

                    reactants, reagents, products = row["rxn_smiles"].split(">")
                    Chem.MolFromSmiles(reactants)       # simply ensures that SMILES can be parsed
                    Chem.MolFromSmiles(products)        # simply ensures that SMILES can be parsed

        logging.info("Data format check passed")

    def preprocess(self) -> None:
        """Actual file-based preprocessing"""
        self.split_src_tgt(canonicalized=True)
        onmt_preprocess(self.model_args)

    def split_src_tgt(self, canonicalized: bool = True):
        """Split reaction SMILES into source and target"""
        logging.info("Splitting reaction SMILES into source and target")
        for phase, fn in [("train", self.train_file),
                          ("val", self.val_file),
                          ("test", self.test_file)]:
            ofn_src = os.path.join(self.processed_data_path, f"src-{phase}.txt")
            ofn_tgt = os.path.join(self.processed_data_path, f"tgt-{phase}.txt")
            invalid_count = 0
            with open(fn, "r") as f, open(ofn_src, "w") as of_src, open(ofn_tgt, "w") as of_tgt:
                csv_reader = csv.DictReader(f)
                for row in tqdm(csv_reader):
                    try:
                        reactants, reagents, products = row["rxn_smiles"].split(">")
                        mols_r = Chem.MolFromSmiles(reactants)
                        mols_p = Chem.MolFromSmiles(products)
                        if mols_r is None or mols_p is None:
                            invalid_count += 1
                            continue

                        [a.ClearProp('molAtomMapNumber') for a in mols_r.GetAtoms()]
                        [a.ClearProp('molAtomMapNumber') for a in mols_p.GetAtoms()]

                        cano_smi_r = Chem.MolToSmiles(mols_r, isomericSmiles=True, canonical=canonicalized)
                        cano_smi_p = Chem.MolToSmiles(mols_p, isomericSmiles=True, canonical=canonicalized)

                        of_src.write(f"{smi_tokenizer(cano_smi_p.strip())}\n")
                        of_tgt.write(f"{smi_tokenizer(cano_smi_r.strip())}\n")
                    except Exception as e:
                        logging.info(e)
                        logging.info(row["rxn_smiles"].split(">"))
                        invalid_count += 1

            logging.info(f"Invalid count: {invalid_count}")
