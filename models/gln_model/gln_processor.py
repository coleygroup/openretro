import csv
import logging
import multiprocessing
import os
import pickle as cp
import random
from base.processor_base import Processor
from collections import Counter, defaultdict
from gln.common.mol_utils import cano_smarts, cano_smiles, smarts_has_useless_parentheses
from gln.data_process.build_raw_template import get_tpl
from gln.data_process.find_centers import find_edges
from rdkit import Chem
from tqdm import tqdm
from typing import Dict, List


class GLNProcessor(Processor):
    """Class for GLN Preprocessing"""

    def __init__(self,
                 model_name: str,
                 config: Dict[str, any],
                 raw_data_files: List[str],
                 processed_data_path: str,
                 num_cores: int = None):
        super().__init__(model_name=model_name,
                         config=config,
                         raw_data_files=raw_data_files,
                         processed_data_path=processed_data_path)
        self.check_count = 100
        self.train_file, self.val_file, self.test_file = raw_data_files
        self.num_cores = num_cores

        self.tpl_folder = None

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
                    if i == 0:                          # header
                        continue
                    if i > self.check_count:            # check the first few rows
                        break

                    assert (c in row for c in ["class", "reactants>reagents>production"]), \
                        f"Error processing file {fn} line {i}, ensure columns 'class' and " \
                        f"'reactants>reagents>production' is included!"
                    assert row["class"] == "UNK" or row["class"].isnumeric(), \
                        f"Error processing file {fn} line {i}, ensure 'class' is UNK or numeric!"

                    reactants, reagents, products = row["reactants>reagents>production"].split(">")
                    Chem.MolFromSmiles(reactants)       # simply ensures that SMILES can be parsed
                    Chem.MolFromSmiles(products)        # simply ensures that SMILES can be parsed

        logging.info("Data format check passed")

    def preprocess(self) -> None:
        """Actual file-based preprocessing"""
        self.get_canonical_smiles()                 # step 0.0
        self.build_raw_template()                   # step 0.1
        self.filter_template()                      # step 1
        self.get_canonical_smarts()                 # step 2
        self.find_centers()                         # step 3
        self.build_all_reactions()                  # step 4

    def get_canonical_smiles(self):
        """Core of step0.0_run_get_cano_smiles.sh, adapted from get_canonical_smiles.py"""
        logging.info(f"Step 0.0: getting canonical SMILES")
        # Reading SMILES
        rxn_smiles = []
        for fn in self.raw_data_files:
            if not fn:
                continue

            with open(fn, "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                rxn_idx = header.index("reactants>reagents>production")
                for row in tqdm(reader):
                    rxn_smiles.append(row[rxn_idx])

        # Canonizing SMILES and creating atom lists
        all_symbols = set()
        smiles_cano_map = {}
        for rxn in tqdm(rxn_smiles):
            reactants, _, prod = rxn.split(">")
            mols = reactants.split(".") + [prod]
            for sm in mols:
                m, cano_sm = cano_smiles(sm)
                if m is not None:
                    for a in m.GetAtoms():
                        all_symbols.add((a.GetAtomicNum(), a.GetSymbol()))
                if sm in smiles_cano_map:
                    assert smiles_cano_map[sm] == cano_sm
                else:
                    smiles_cano_map[sm] = cano_sm
        logging.info(f"num of smiles: {len(smiles_cano_map)}")

        set_mols = set()
        for s in smiles_cano_map:
            set_mols.add(smiles_cano_map[s])
        logging.info(f"# unique smiles: {len(set_mols)}")

        with open(os.path.join(self.processed_data_path, "cano_smiles.pkl"), "wb") as f:
            cp.dump(smiles_cano_map, f, cp.HIGHEST_PROTOCOL)
        logging.info(f"# unique atoms: {len(all_symbols)}")

        all_symbols = sorted(list(all_symbols))
        with open(os.path.join(self.processed_data_path, "atom_list.txt"), "w") as f:
            for a in all_symbols:
                f.write("%d\n" % a[0])

    @staticmethod
    def get_writer(fname: str, header: List[str]):
        fout = open(fname, "w")
        writer = csv.writer(fout)
        writer.writerow(header)

        return fout, writer

    def build_raw_template(self):
        """Core of step0.1_run_template_extract.sh, adapted from build_raw_template.py"""
        logging.info(f"Step 0.1: building raw templates")
        with open(self.train_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [row for row in reader]

        pool = multiprocessing.Pool(self.num_cores)
        tasks = []
        for idx, row in tqdm(enumerate(rows)):
            row_idx, _, rxn_smiles = row
            tasks.append((idx, row_idx, rxn_smiles))

        fn = os.path.join(self.processed_data_path, "proc_train_singleprod.csv")
        fn_failed = os.path.join(self.processed_data_path, "failed_template.csv")
        fout, writer = self.get_writer(fn, ["id", "class", "rxn_smiles", "retro_templates"])
        fout_failed, failed_writer = self.get_writer(fn_failed, ["id", "class", "rxn_smiles", "err_msg"])

        for result in tqdm(pool.imap_unordered(get_tpl, tasks), total=len(tasks)):
            idx, template = result
            row_idx, rxn_type, rxn_smiles = rows[idx]

            if "reaction_smarts" in template:
                writer.writerow([row_idx, rxn_type, rxn_smiles, template["reaction_smarts"]])
                fout.flush()
            else:
                failed_writer.writerow([row_idx, rxn_type, rxn_smiles, template["err_msg"]])
                fout_failed.flush()

        fout.close()
        fout_failed.close()

        pool.close()
        pool.join()

    def filter_template(self):
        """Core of step1_filter_template.sh, adapted from filter_template.py"""
        logging.info(f"Step 1: filtering templates")
        proc_file = os.path.join(self.processed_data_path, "proc_train_singleprod.csv")

        unique_tpls = Counter()
        tpl_types = defaultdict(set)
        with open(proc_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            logging.info(f"Header: {header}")
            for row in tqdm(reader):
                tpl = row[header.index("retro_templates")]
                rxn_type = row[header.index("class")]
                tpl_types[tpl].add(rxn_type)
                unique_tpls[tpl] += 1

        logging.info(f"total # templates: {len(unique_tpls)}")

        used_tpls = []
        for x in unique_tpls:
            if unique_tpls[x] >= self.config["tpl_min_cnt"]:
                used_tpls.append(x)
        logging.info(f"num templates after filtering: {len(used_tpls)}")

        self.tpl_folder = os.path.join(self.processed_data_path, "tpl-default")
        os.makedirs(self.tpl_folder, exist_ok=True)

        out_file = os.path.join(self.tpl_folder, "templates.csv")
        with open(out_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "retro_templates"])
            for x in used_tpls:
                for t in tpl_types[x]:
                    writer.writerow([t, x])

    def get_canonical_smarts(self):
        """Core of step2_run_get_cano_smarts.sh, adapted from get_canonical_smarts.py"""
        logging.info(f"Step 2: getting canonical SMARTS")
        # Reading templates
        tpl_file = os.path.join(self.tpl_folder, "templates.csv")

        retro_templates = []
        with open(tpl_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)

            for row in tqdm(reader):
                retro_templates.append(row[header.index("retro_templates")])

        # Canonizing templates
        prod_cano_smarts = set()
        react_cano_smarts = set()

        smarts_cano_map = {}
        pbar = tqdm(retro_templates)
        for template in pbar:
            sm_prod, _, sm_react = template.split(">")
            if smarts_has_useless_parentheses(sm_prod):
                sm_prod = sm_prod[1:-1]

            smarts_cano_map[sm_prod] = cano_smarts(sm_prod)[1]
            prod_cano_smarts.add(smarts_cano_map[sm_prod])

            for r_smarts in sm_react.split('.'):
                smarts_cano_map[r_smarts] = cano_smarts(r_smarts)[1]
                react_cano_smarts.add(smarts_cano_map[r_smarts])
            pbar.set_description(
                "# prod centers: %d, # react centers: %d" % (len(prod_cano_smarts), len(react_cano_smarts)))
        logging.info("# prod centers: %d, # react centers: %d" % (len(prod_cano_smarts), len(react_cano_smarts)))

        # Saving
        with open(os.path.join(self.tpl_folder, "prod_cano_smarts.txt"), "w") as f:
            for s in prod_cano_smarts:
                f.write('%s\n' % s)
        with open(os.path.join(self.tpl_folder, "react_cano_smarts.txt"), "w") as f:
            for s in react_cano_smarts:
                f.write('%s\n' % s)
        with open(os.path.join(self.tpl_folder, "cano_smarts.pkl"), "wb") as f:
            cp.dump(smarts_cano_map, f, cp.HIGHEST_PROTOCOL)

    def find_centers(self):
        """Core of step3_run_find_centers.sh, adapted from find_centers.py"""
        logging.info(f"Step 3: finding reaction centers")
        with open(os.path.join(self.processed_data_path, "cano_smiles.pkl"), "rb") as f:
            smiles_cano_map = cp.load(f)
        with open(os.path.join(self.tpl_folder, "cano_smarts.pkl"), "rb") as f:
            smarts_cano_map = cp.load(f)
        with open(os.path.join(self.tpl_folder, "prod_cano_smarts.txt"), "r") as f:
            prod_cano_smarts = [row.strip() for row in f.readlines()]

        prod_center_mols = []
        for sm in tqdm(prod_cano_smarts):
            prod_center_mols.append((sm, Chem.MolFromSmarts(sm)))

        logging.info(f"num of prod centers: {len(prod_center_mols)}")
        logging.info(f"num of smiles: {len(smiles_cano_map)}")

        csv_file = os.path.join(self.processed_data_path, "templates.csv")

        smarts_type_set = defaultdict(set)
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            tpl_idx = header.index("retro_templates")
            type_idx = header.index("class")
            for row in reader:
                rxn_type = row[type_idx]
                template = row[tpl_idx]
                sm_prod, _, _ = template.split(">")
                if smarts_has_useless_parentheses(sm_prod):
                    sm_prod = sm_prod[1:-1]
                sm_prod = smarts_cano_map[sm_prod]
                smarts_type_set[sm_prod].add(rxn_type)

        if self.config["num_parts"] <= 0:
            num_parts = self.num_cores
        else:
            num_parts = self.config["num_parts"]

        pool = multiprocessing.Pool(self.num_cores)

        for out_phase, csv_file in [("train", self.train_file),
                                    ("val", self.val_file),
                                    ("test", self.test_file)]:
            if not csv_file:
                continue

            rxn_smiles = []
            with open(csv_file, "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                rxn_idx = header.index("reactants>reagents>production")
                type_idx = header.index("class")
                for row in tqdm(reader):
                    rxn_smiles.append((row[type_idx], row[rxn_idx]))

            part_size = min(len(rxn_smiles) // num_parts + 1, len(rxn_smiles))

            for pid in range(num_parts):
                idx_range = range(pid * part_size, min((pid + 1) * part_size, len(rxn_smiles)))
                local_results = [[None, None, None] for _ in idx_range]     # do not use * N for list

                tasks = []
                for i, idx in enumerate(idx_range):
                    rxn_type, rxn = rxn_smiles[idx]
                    reactants, _, prod = rxn.split(">")
                    tasks.append((i, rxn_type, prod))
                for result in tqdm(pool.imap_unordered(find_edges, tasks), total=len(tasks)):
                    i, rxn_type, smiles, centers = result
                    local_results[i] = (rxn_type, smiles, centers)

                out_folder = os.path.join(self.tpl_folder, f"np-{num_parts}")
                os.makedirs(out_folder, exist_ok=True)
                fout = open(os.path.join(out_folder, f"{out_phase}-prod_center_maps-part-{pid}.csv"), "w")
                writer = csv.writer(fout)
                writer.writerow(["smiles", "class", "centers"])

                for i in range(len(local_results)):
                    rxn_type, smiles, centers = local_results[i]
                    if centers is not None:
                        writer.writerow([smiles, rxn_type, centers])
                fout.close()

        pool.close()
        pool.join()

    def build_all_reactions(self):
        """Core of step4_run_find_all_reactions.sh, adapted from build_all_reactions.py"""
        logging.info(f"Step 4: building all reactions")
        random.seed(self.config["seed"])

        DataInfo.init(cmd_args.dropbox, cmd_args)

        fn_pos = lambda idx: get_writer('pos_tpls-part-%d.csv' % idx,
                                        ['tpl_idx', 'pos_tpl_idx', 'num_tpl_compete', 'num_react_compete'])
        fn_neg = lambda idx: get_writer('neg_reacts-part-%d.csv' % idx, ['sample_idx', 'neg_reactants'])

        if cmd_args.num_parts <= 0:
            num_parts = cmd_args.num_cores
            DataInfo.load_cooked_part('train', load_graphs=False)
        else:
            num_parts = cmd_args.num_parts

        train_reactions = load_train_reactions(cmd_args)
        n_train = len(train_reactions)
        part_size = n_train // num_parts + 1

        if cmd_args.part_num > 0:
            prange = range(cmd_args.part_id, cmd_args.part_id + cmd_args.part_num)
        else:
            prange = range(num_parts)
        for pid in prange:
            f_pos, writer_pos = fn_pos(pid)
            f_neg, writer_neg = fn_neg(pid)
            if cmd_args.num_parts > 0:
                DataInfo.load_cooked_part('train', part=pid, load_graphs=False)
            part_tasks = []
            idx_range = list(range(pid * part_size, min((pid + 1) * part_size, n_train)))
            for i in idx_range:
                part_tasks.append((i, train_reactions[i]))

            pool = multiprocessing.Pool(cmd_args.num_cores)
            for result in tqdm(pool.imap_unordered(find_tpls, part_tasks), total=len(idx_range)):
                if result is None:
                    continue
                idx, pos_tpl_idx, neg_reactions = result
                idx = str(idx)
                neg_keys = neg_reactions

                if cmd_args.max_neg_reacts > 0:
                    neg_keys = list(neg_keys)
                    random.shuffle(neg_keys)
                    neg_keys = neg_keys[:cmd_args.max_neg_reacts]
                for pred in neg_keys:
                    writer_neg.writerow([idx, pred])
                for key in pos_tpl_idx:
                    nt, np = pos_tpl_idx[key]
                    writer_pos.writerow([idx, key, nt, np])
                f_pos.flush()
                f_neg.flush()
            f_pos.close()
            f_neg.close()
            pool.close()
            pool.join()
