import csv
import logging
import multiprocessing
import numpy as np
import os
import pickle
import scipy
import sys
from base.processor_base import Processor
from concurrent.futures import TimeoutError
from functools import partial
from pebble import ProcessPool
from rdchiral.template_extractor import extract_from_reaction
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from scipy import sparse
from tqdm import tqdm
from typing import Dict, List


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


def gen_prod_fps_helper(args, rxn_smi):
    prod_smi_map = rxn_smi.split('>>')[-1]
    prod_mol = Chem.MolFromSmiles(prod_smi_map)
    if prod_mol is None:
        logging.info(rxn_smi)
    [atom.ClearProp('molAtomMapNumber') for atom in prod_mol.GetAtoms()]
    prod_smi_nomap = Chem.MolToSmiles(prod_mol, True)
    # Sometimes stereochem takes another canonicalization... (just in case)
    prod_smi_nomap = Chem.MolToSmiles(Chem.MolFromSmiles(prod_smi_nomap), True)

    prod_fp = mol_smi_to_count_fp(prod_smi_nomap, args.radius, args.fp_size)
    return prod_smi_nomap, prod_fp


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


def get_template_idx(temps_dict, task):
    rxn_idx, r, p = task
    ############################################################
    # original label generation pipeline
    # extract template for this rxn_smi, and match it to template dictionary from training data
    rxn = (rxn_idx, r, p)   # r & p must be atom-mapped
    rxn_idx, rxn_template = get_tpl(task)

    if rxn_template is None or 'reaction_smarts' not in rxn_template:
        return rxn_idx, -1                  # unable to extract template
    p_temp = cano_smarts(rxn_template['products'])
    r_temp = cano_smarts(rxn_template['reactants'])
    cano_temp = p_temp + '>>' + r_temp

    if cano_temp in temps_dict:
        return rxn_idx, temps_dict[cano_temp]
    else:
        return rxn_idx, len(temps_dict)     # no template matching


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
        self.check_count = 100
        self.train_file, self.val_file, self.test_file = raw_data_files
        self.num_cores = num_cores

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
        """Actual file-based preprocessing, adpated from prepare_data.py"""
        self.gen_prod_fps()
        self.variance_cutoff()
        self.get_train_templates()
        self.match_templates()

    def gen_prod_fps(self):
        # parallelizing makes it very slow for some reason
        # ~2 min on 40k train prod_smi on 16 cores for 32681-dim
        for phase, fn in [("train", self.train_file),
                          ("val", self.val_file),
                          ("test", self.test_file)]:
            logging.info(f'Processing {phase}')

            with open(fn, "r") as csv_file:
                csv_reader = csv.DictReader(csv_file)
                clean_rxnsmi_phase = [row["rxn_smiles"].strip()
                                      for row in csv_reader]

            logging.info(f'Parallelizing over {self.num_cores} cores')
            pool = multiprocessing.Pool(self.num_cores)

            phase_prod_smi_nomap = []
            phase_rxn_prod_fps = []
            gen_prod_fps_partial = partial(gen_prod_fps_helper, self.model_args)
            for result in tqdm(pool.imap(gen_prod_fps_partial, clean_rxnsmi_phase),
                               total=len(clean_rxnsmi_phase),
                               desc='Processing rxn_smi'):
                prod_smi_nomap, prod_fp = result
                phase_prod_smi_nomap.append(prod_smi_nomap)
                phase_rxn_prod_fps.append(prod_fp)

            pool.close()
            pool.join()

            # these are the input data into the network
            phase_rxn_prod_fps = sparse.vstack(phase_rxn_prod_fps)
            sparse.save_npz(
                os.path.join(self.processed_data_path, f"prod_fps_{phase}.npz"),
                phase_rxn_prod_fps
            )

            with open(os.path.join(self.processed_data_path, f"prod_smis_nomap_{phase}.smi"), "wb") as of:
                pickle.dump(phase_prod_smi_nomap, of, protocol=4)

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

    def get_train_templates(self):
        """
        For the expansion rules, a more general rule definition was employed. Here, only
        the reaction centre was extracted. Rules occurring at least three times
        were kept. The two sets encompass 17,134 and 301,671 rules, and cover
        52% and 79% of all chemical reactions from 2015 and after, respectively.
        """
        # ~40 sec on 40k train rxn_smi on 16 cores
        logging.info('Extracting templates from training data')

        with open(self.train_file, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            clean_rxnsmi_phase = [row["rxn_smiles"].strip()
                                  for row in csv_reader]

        templates = {}
        rxns = []
        for idx, rxn_smi in enumerate(clean_rxnsmi_phase):
            r = rxn_smi.split('>>')[0]
            p = rxn_smi.split('>>')[-1]
            rxns.append((idx, r, p))
        logging.info(f'Total training rxns: {len(rxns)}')

        logging.info(f'Parallelizing over {self.num_cores} cores')
        invalid_temp = 0
        # here the order doesn't matter since we just want a dictionary of templates
        with ProcessPool(max_workers=self.num_cores) as pool:
            # had to resort to pebble to add timeout. rdchiral could hang
            future = pool.map(get_tpl, rxns, timeout=10)

            iterator = future.result()
            while True:
                try:
                    result = next(iterator)
                    idx, template = result

                    if idx % 10000 == 0:
                        logging.info(f"Processing {idx}th reaction")

                    if template is None or 'reaction_smarts' not in template:
                        invalid_temp += 1
                        # have to comment out this.. there might be a lot of such reactions!
                        # logging.info(f'At {idx}, could not extract template')
                        continue  # no template could be extracted

                    # canonicalize template (needed, bcos q a number of templates are equivalent, 10247 --> 10198)
                    p_temp = cano_smarts(template['products'])
                    r_temp = cano_smarts(template['reactants'])
                    cano_temp = p_temp + '>>' + r_temp
                    # NOTE: 'reaction_smarts' is actually: p_temp >> r_temp !!!!!

                    if cano_temp not in templates:
                        templates[cano_temp] = 1
                    else:
                        templates[cano_temp] += 1
                except StopIteration:
                    break
                except TimeoutError as error:
                    logging.info(f"get_tpl call took more than {error.args} seconds")

        # for result in tqdm(pool.imap_unordered(get_tpl, rxns),
        #                    total=len(rxns)):
        #     idx, template = result
        #     if template is None or 'reaction_smarts' not in template:
        #         invalid_temp += 1
        #         # have to comment out this.. there might be a lot of such reactions!
        #         # logging.info(f'At {idx}, could not extract template')
        #         continue  # no template could be extracted
        #
        #     # canonicalize template (needed, bcos q a number of templates are equivalent, 10247 --> 10198)
        #     p_temp = cano_smarts(template['products'])
        #     r_temp = cano_smarts(template['reactants'])
        #     cano_temp = p_temp + '>>' + r_temp
        #     # NOTE: 'reaction_smarts' is actually: p_temp >> r_temp !!!!!
        #
        #     if cano_temp not in templates:
        #         templates[cano_temp] = 1
        #     else:
        #         templates[cano_temp] += 1

        pool.close()
        pool.join()

        logging.info(f'No of rxn where template extraction failed: {invalid_temp}')

        templates = sorted(templates.items(), key=lambda x: x[1], reverse=True)
        templates = [f"{p[0]}: {p[1]}\n" for p in templates]
        with open(os.path.join(self.processed_data_path, "training_templates.txt"), "w") as of:
            of.writelines(templates)

    def match_templates(self):
        # ~3-4 min on 40k train rxn_smi on 16 cores
        template_file = os.path.join(self.processed_data_path, "training_templates.txt")
        logging.info(f'Loading templates from file: {template_file}')

        with open(template_file, "r") as f:
            lines = f.readlines()
        temps_filtered = []
        temps_dict = {}  # build mapping from temp to idx for O(1) find
        temps_idx = 0
        for l in lines:
            pa, cnt = l.strip().split(': ')
            if int(cnt) >= self.model_args.min_freq:
                temps_filtered.append(pa)
                temps_dict[pa] = temps_idx
                temps_idx += 1
        logging.info(f'Total number of template patterns: {len(temps_filtered)}')

        logging.info(f'Parallelizing over {self.num_cores} cores')
        # pool = multiprocessing.Pool(self.num_cores)

        logging.info('Matching against extracted templates')
        for phase, fn in [("train", self.train_file),
                          ("val", self.val_file),
                          ("test", self.test_file)]:
            logging.info(f'Processing {phase}')
            with open(fn, "r") as csv_file:
                csv_reader = csv.DictReader(csv_file)
                clean_rxnsmi_phase = [row["rxn_smiles"].strip()
                                      for row in csv_reader]

            with open(os.path.join(self.processed_data_path, f"prod_smis_nomap_{phase}.smi"), 'rb') as f:
                phase_prod_smi_nomap = pickle.load(f)

            tasks = []
            for idx, rxn_smi in tqdm(enumerate(clean_rxnsmi_phase), desc='Building tasks',
                                     total=len(clean_rxnsmi_phase)):
                r = rxn_smi.split('>>')[0]
                p = rxn_smi.split('>>')[1]
                tasks.append((idx, r, p))

            # make CSV file to save labels (template_idx) & rxn data for monitoring training
            col_names = ['rxn_idx', 'prod_smi', 'rcts_smi', 'temp_idx', 'template']
            rows = []
            labels = []
            found = 0
            get_template_partial = partial(get_template_idx, temps_dict)
            # don't use imap_unordered!!!! it doesn't guarantee the order, or we can use it and then sort by rxn_idx
            with ProcessPool(max_workers=self.num_cores) as pool:
                # had to resort to pebble to add timeout. rdchiral could hang
                future = pool.map(get_template_partial, tasks, timeout=10)

                iterator = future.result()
                while True:
                    try:
                        result = next(iterator)

                        rxn_idx, template_idx = result
                        if rxn_idx % 10000 == 0:
                            logging.info(f"Processing {rxn_idx}th reaction")

                        rcts_smi_map = clean_rxnsmi_phase[rxn_idx].split('>>')[0]
                        rcts_mol = Chem.MolFromSmiles(rcts_smi_map)
                        [atom.ClearProp('molAtomMapNumber') for atom in rcts_mol.GetAtoms()]
                        rcts_smi_nomap = Chem.MolToSmiles(rcts_mol, True)
                        # Sometimes stereochem takes another canonicalization...
                        rcts_smi_nomap = Chem.MolToSmiles(Chem.MolFromSmiles(rcts_smi_nomap), True)

                        template = temps_filtered[template_idx] if template_idx != len(temps_filtered) else ''
                        rows.append([
                            rxn_idx,
                            phase_prod_smi_nomap[rxn_idx],
                            rcts_smi_nomap,  # tasks[rxn_idx][1],
                            template,
                            template_idx,
                        ])
                        labels.append(template_idx)
                        found += (template_idx != len(temps_filtered))
                    except StopIteration:
                        break
                    except TimeoutError as error:
                        logging.info(f"get_tpl call took more than {error.args} seconds")

            logging.info(f'Template coverage: {found / len(tasks) * 100:.2f}%')
            labels = np.array(labels)
            np.save(os.path.join(self.processed_data_path, f"labels_{phase}.npy"), labels)

            ofn = os.path.join(self.processed_data_path, f"processed_{phase}.csv")
            with open(ofn, "w") as out_csv:
                writer = csv.writer(out_csv)
                writer.writerow(col_names)  # header
                for row in rows:
                    writer.writerow(row)

        pool.close()
        pool.join()
