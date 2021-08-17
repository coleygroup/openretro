import csv
import logging
import multiprocessing
import numpy as np
import os
import pickle
import scipy
import sys
import time
from base.processor_base import Processor
from utils.chem_utils import canonicalize_smiles
from concurrent.futures import TimeoutError
from functools import partial
from pebble import ProcessPool
from rdchiral.template_extractor import extract_from_reaction
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from scipy import sparse
from tqdm import tqdm
from typing import Dict, List, Tuple

global G_temps_dict


class BlockPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


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


def gen_prod_fps_helper(args, rxn_smi) -> Tuple[str, str, scipy.sparse.csr_matrix]:
    r_smi_map, p_smi_map = rxn_smi.split('>>')
    r_smi_nomap = canonicalize_smiles(r_smi_map, remove_atom_number=True)
    p_smi_nomap = canonicalize_smiles(p_smi_map, remove_atom_number=True)
    if not r_smi_nomap:
        logging.info(f"empty cano reactants for rxn: {rxn_smi}")
    if not p_smi_nomap:
        logging.info(f"empty cano product for rxn: {rxn_smi}")
    prod_fp = mol_smi_to_count_fp(p_smi_nomap, args.radius, args.fp_size)

    return r_smi_nomap, p_smi_nomap, prod_fp


def log_row(row):
    return sparse.csr_matrix(np.log(row.toarray() + 1))


def var_col(col):
    return np.var(col.toarray())


def pass_bond_edits_test(r: str, p: str, max_rbonds=5, max_pbonds=3, max_atoms=10):
    """Adapted from filter.py and rdkit.py in temprel"""
    rmol = Chem.MolFromSmiles(r)
    pmol = Chem.MolFromSmiles(p)

    pbonds = []
    for bond in pmol.GetBonds():
        a = bond.GetBeginAtom().GetAtomMapNum()
        b = bond.GetEndAtom().GetAtomMapNum()
        if a or b:
            pbonds.append(tuple(sorted([a, b])))

    rbonds = []
    for bond in rmol.GetBonds():
        a = bond.GetBeginAtom().GetAtomMapNum()
        b = bond.GetEndAtom().GetAtomMapNum()
        if a or b:
            rbonds.append(tuple(sorted([a, b])))

    r_changed = set(rbonds) - set(pbonds)
    p_changed = set(pbonds) - set(rbonds)

    if len(r_changed) > max_rbonds or len(p_changed) > max_pbonds:
        return False

    atoms_changed = set()
    for ch in list(r_changed) + list(p_changed):
        atoms_changed.add(ch[0])
        atoms_changed.add(ch[1])
    atoms_changed -= {0}

    if len(atoms_changed) > max_atoms:
        return False

    # if passing all three criteria
    return True


def get_tpl(task):
    idx, react, prod = task
    if not pass_bond_edits_test(r=react, p=prod):
        return idx, None

    reaction = {'_id': idx, 'reactants': react, 'products': prod}
    try:
        with BlockPrint():
            template = extract_from_reaction(reaction)
        # https://github.com/connorcoley/rdchiral/blob/master/rdchiral/template_extractor.py
    except:
        return idx, None
    return idx, template


def cano_smarts(smarts):
    tmp = Chem.MolFromSmarts(smarts)
    if tmp is None:
        logging.info(f'Could not parse {smarts}')
        return smarts
    # do not remove atom map number
    # [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
    cano = Chem.MolToSmarts(tmp)
    if '[[se]]' in cano:  # strange parse error
        cano = smarts
    return cano


def get_template_idx(task: str):
    global G_temps_dict
    temps_dict = G_temps_dict

    rxn_idx, cano_temp = task.strip().split("\t")
    if cano_temp == ">>":
        temp_idx = 0
    elif cano_temp in temps_dict:
        temp_idx = temps_dict[cano_temp]
    else:
        temp_idx = 0

    if temp_idx == 0:
        cano_temp = ""

    return int(rxn_idx), temp_idx, cano_temp


class NeuralSymProcessor(Processor):
    """Class for NeuralSym Preprocessing"""

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
        self.train_file, self.val_file, self.test_file = raw_data_files
        self.num_cores = num_cores

    def preprocess(self) -> None:
        """Actual file-based preprocessing, adapted from prepare_data.py"""
        self.gen_prod_fps()
        self.variance_cutoff()
        self.get_templates()
        self.filter_templates_by_freq()

    def gen_prod_fps(self):
        # parallelizing makes it very slow for some reason
        # ~2 min on 40k train prod_smi on 16 cores for 32681-dim
        for phase, fn in [("train", self.train_file),
                          ("val", self.val_file),
                          ("test", self.test_file)]:
            logging.info(f'Processing {phase}')

            with open(fn, "r") as csv_file:
                csv_reader = csv.DictReader(csv_file)
                clean_rxnsmi_phase = [row["rxn_smiles"].strip() for row in csv_reader]

            logging.info(f'Parallelizing over {self.num_cores} cores')
            pool = multiprocessing.Pool(self.num_cores)

            phase_rxn_prod_fps = []
            gen_prod_fps_partial = partial(gen_prod_fps_helper, self.model_args)

            ofn = os.path.join(self.processed_data_path, f"cano_smis_nomap_{phase}.txt")
            with open(ofn, "w") as of:
                for result in tqdm(pool.imap(gen_prod_fps_partial, clean_rxnsmi_phase),
                                   total=len(clean_rxnsmi_phase),
                                   desc='Processing rxn_smi'):
                    r_smi_nomap, p_smi_nomap, prod_fp = result
                    phase_rxn_prod_fps.append(prod_fp)

                    cano_smi_nomap = f"{r_smi_nomap}>>{p_smi_nomap}\n"
                    of.write(cano_smi_nomap)

            pool.close()
            pool.join()

            # these are the input data into the network
            phase_rxn_prod_fps = sparse.vstack(phase_rxn_prod_fps)
            sparse.save_npz(
                os.path.join(self.processed_data_path, f"prod_fps_{phase}.npz"),
                phase_rxn_prod_fps
            )

    def variance_cutoff(self):
        # for training dataset (40k rxn_smi):
        # ~1 min to do log(x+1) transformation on 16 cores, and then
        # ~2 min to gather variance statistics across 1 million indices on 16 cores, and then
        # ~5 min to build final 32681-dim fingerprint on 16 cores
        logging.info(f'Parallelizing over {self.num_cores} cores')
        pool = multiprocessing.Pool(self.num_cores)

        for phase in ["train", "val", "test"]:
            prod_fps = sparse.load_npz(os.path.join(self.processed_data_path, f"prod_fps_{phase}.npz"))

            logged = []
            # imap is much, much faster than map
            # take log(x+1), ~2.5 min for 1mil-dim on 8 cores (parallelized)
            for result in tqdm(pool.imap(log_row, prod_fps),
                               total=prod_fps.shape[0],
                               desc='Taking log(x+1)'):
                logged.append(result)
            logged = sparse.vstack(logged)

            # collect variance statistics by column index from training product fingerprints
            # VERY slow with 2 for-loops to access each element individually.
            # idea: transpose the sparse matrix, then go through 1 million rows using pool.imap
            # massive speed up from 280 hours to 1 hour on 8 cores
            logged = logged.transpose()  # [39713, 1 mil] -> [1 mil, 39713]

            if phase == "train":
                # no need to store all the values from each col_idx (results in OOM).
                # just calc variance immediately, and move on
                variances = []
                # imap directly on csr_matrix is the fastest!!! from 1 hour --> ~2 min on 20 cores (parallelized)
                for result in tqdm(pool.imap(var_col, logged),
                                   total=logged.shape[0],
                                   desc='Collecting fingerprint values by indices'):
                    variances.append(result)

                indices_ordered = list(range(logged.shape[0]))      # should be 1,000,000
                indices_ordered.sort(key=lambda x: variances[x], reverse=True)

                # need to save sorted indices for infer_one API
                indices_np = np.array(indices_ordered[:self.model_args.final_fp_size])
                np.savetxt(os.path.join(self.processed_data_path, "variance_indices.txt"), indices_np)

            logged = logged.transpose()  # [1 mil, 39713] -> [39713, 1 mil]
            # build and save final thresholded fingerprints
            thresholded = []
            for row_idx in tqdm(range(logged.shape[0]), desc='Building thresholded fingerprints'):
                thresholded.append(
                    logged[row_idx, indices_ordered[:self.model_args.final_fp_size]]  # should be 32,681
                )
            thresholded = sparse.vstack(thresholded)
            sparse.save_npz(
                os.path.join(self.processed_data_path, f"to_{self.model_args.final_fp_size}_prod_fps_{phase}.npz"),
                thresholded
            )

        pool.close()
        pool.join()

    def get_templates(self):
        """
        For the expansion rules, a more general rule definition was employed. Here, only
        the reaction centre was extracted. Rules occurring at least three times
        were kept. The two sets encompass 17,134 and 301,671 rules, and cover
        52% and 79% of all chemical reactions from 2015 and after, respectively.
        """
        # ~40 sec on 40k train rxn_smi on 16 cores
        start = time.time()
        templates = {}

        for phase, fn in [("train", self.train_file),
                          ("val", self.val_file),
                          ("test", self.test_file)]:
            logging.info(f"Extracting templates from phase {phase}")

            with open(fn, "r") as csv_file:
                csv_reader = csv.DictReader(csv_file)
                clean_rxnsmi_phase = [row["rxn_smiles"].strip() for row in csv_reader]

            rxns = []
            for idx, rxn_smi in enumerate(clean_rxnsmi_phase):
                r = rxn_smi.split('>>')[0]
                p = rxn_smi.split('>>')[-1]
                rxns.append((idx, r, p))
            logging.info(f'Total rxns for phase {phase}: {len(rxns)}')

            logging.info(f'Parallelizing over {self.num_cores} cores')
            invalid_temp = 0

            ofn = os.path.join(self.processed_data_path, f"{phase}_templates_by_idx.txt")
            with ProcessPool(max_workers=self.num_cores) as pool, open(ofn, "w") as of:
                # had to resort to pebble to add timeout. rdchiral could hang
                future = pool.map(get_tpl, rxns, timeout=10)

                iterator = future.result()
                while True:
                    try:
                        result = next(iterator)
                        idx, template = result

                        if idx % 10000 == 0:
                            logging.info(f"Processing {idx}th reaction, elapsed time: {time.time() - start}")

                        # no template could be extracted
                        if template is None or 'reaction_smarts' not in template:
                            invalid_temp += 1
                            # have to comment out this.. there might be a lot of such reactions!
                            # logging.info(f'At {idx}, could not extract template')

                            cano_temp = "failed_extract"

                        else:
                            # canonicalize template
                            p_temp = cano_smarts(template['products'])
                            r_temp = cano_smarts(template['reactants'])

                            # NOTE: 'reaction_smarts' is actually: p_temp >> r_temp !!!!!
                            cano_temp = p_temp + '>>' + r_temp

                            if phase == "train":
                                if cano_temp in templates:
                                    templates[cano_temp] += 1
                                else:
                                    templates[cano_temp] = 1
                        of.write(f"{idx}\t{cano_temp}\n")

                    except StopIteration:
                        break
                    except TimeoutError as error:
                        logging.info(f"get_tpl call took more than {error.args} seconds")

            pool.close()
            pool.join()

            logging.info(f'No of rxn where template extraction failed: {invalid_temp}')

        templates = sorted(templates.items(), key=lambda x: x[1], reverse=True)
        templates = ["failed_extract: 99999\n"] + \
                    [f"{p[0]}: {p[1]}\n" for p in templates]
        with open(os.path.join(self.processed_data_path, "training_templates.txt"), "w") as of:
            of.writelines(templates)

    def filter_templates_by_freq(self):
        # ~3-4 min on 40k train rxn_smi on 16 cores
        global G_temps_dict

        template_file = os.path.join(self.processed_data_path, "training_templates.txt")
        logging.info(f'Loading templates from file: {template_file}')

        temps_filtered = []
        temps_dict = {}  # build mapping from temp to idx for O(1) find
        temps_idx = 0
        with open(template_file, "r") as f:
            for line in f:
                template, count = line.strip().split(': ')
                if int(count) < self.model_args.min_freq:
                    break               # since it's been sorted
                temps_filtered.append(template)
                temps_dict[template] = temps_idx
                temps_idx += 1
        logging.info(f'Total number of template patterns: {len(temps_filtered) - 1}')
        G_temps_dict = temps_dict

        logging.info(f'Parallelizing over {self.num_cores} cores')
        p = multiprocessing.Pool(self.num_cores)

        for phase, fn in [("train", self.train_file),
                          ("val", self.val_file),
                          ("test", self.test_file)]:
            logging.info(f"Matching against extracted templates for phase {phase}")

            fn = os.path.join(self.processed_data_path, f"cano_smis_nomap_{phase}.txt")
            with open(fn, "r") as f:
                cano_smis_nomap = f.readlines()

            fn = os.path.join(self.processed_data_path, f"{phase}_templates_by_idx.txt")
            with open(fn, "r") as f:
                templates = f.readlines()

            rows = []
            labels = []
            found = 0
            # get_template_partial = partial(get_template_idx, temps_dict)

            for rxn_idx, temp_idx, template in tqdm(p.imap(get_template_idx, templates)):
                cano_smi_nomap = cano_smis_nomap[rxn_idx]
                rcts_smi_nomap, prod_smi_nomap = cano_smi_nomap.strip().split(">>")

                rows.append([rxn_idx, prod_smi_nomap, rcts_smi_nomap, template, temp_idx])
                labels.append(temp_idx)
                found += (temp_idx > 0)

            logging.info(f'Template coverage: {found / len(templates) * 100: .2f}%')
            labels = np.array(labels)
            np.save(os.path.join(self.processed_data_path, f"labels_{phase}.npy"), labels)

            # make CSV file to save labels (template_idx) & rxn data for monitoring training
            col_names = ['rxn_idx', 'prod_smi', 'rcts_smi' 'template', 'temp_idx', ]
            ofn = os.path.join(self.processed_data_path, f"processed_{phase}.csv")
            with open(ofn, "w") as out_csv:
                writer = csv.writer(out_csv)
                writer.writerow(col_names)          # header
                for row in rows:
                    writer.writerow(row)

        p.close()
        p.join()
