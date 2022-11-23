import csv
import logging
import os
import pandas as pd
from base.processor_base import Processor
from models.localretro_model.Extract_from_train_data import get_full_template
from models.localretro_model.LocalTemplate.template_extractor import extract_from_reaction
from models.localretro_model.Run_preprocessing import get_edit_site_retro
from typing import Dict, List


class LocalRetroProcessor(Processor):
    """Class for LocalRetro Preprocessing"""

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
        self.setting = {'verbose': False, 'use_stereo': True, 'use_symbol': True,
                        'max_unmap': 5, 'retro': True, 'remote': True, 'least_atom_num': 2,
                        "max-edit-n": 8}
        self.RXNHASCLASS = False

    def preprocess(self) -> None:
        """Actual file-based preprocessing"""
        self.extract_templates()
        self.match_templates()

    def extract_templates(self):
        """Adapted from Extract_from_train_data.py"""
        logging.info(f"Extracting templates from {self.train_file}")

        with open(self.train_file, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            rxns = [row["rxn_smiles"].strip() for row in csv_reader]

        TemplateEdits = {}
        TemplateCs, TemplateHs, TemplateSs = {}, {}, {}
        TemplateFreq, templates_A, templates_B = {}, {}, {}

        for i, rxn in enumerate(rxns):
            try:
                rxn = {'reactants': rxn.split('>')[0], 'products': rxn.split('>')[-1], '_id': i}
                result = extract_from_reaction(rxn, self.setting)
                if 'reactants' not in result or 'reaction_smarts' not in result.keys():
                    logging.info(f'\ntemplate problem: id: {i}')
                    continue
                reactant = result['reactants']
                template = result['reaction_smarts']
                edits = result['edits']
                H_change = result['H_change']
                Charge_change = result['Charge_change']
                Chiral_change = result["Chiral_change"] if self.setting["use_stereo"] else {}

                template_H = get_full_template(template, H_change, Charge_change, Chiral_change)
                if template_H not in TemplateHs.keys():
                    TemplateEdits[template_H] = {edit_type: edits[edit_type][2] for edit_type in edits}
                    TemplateHs[template_H] = H_change
                    TemplateCs[template_H] = Charge_change
                    TemplateSs[template_H] = Chiral_change

                TemplateFreq[template_H] += 1
                for edit_type, bonds in edits.items():
                    bonds = bonds[0]
                    if len(bonds) > 0:
                        if edit_type in ['A', 'R']:
                            templates_A[template_H] += 1
                        else:
                            templates_B[template_H] += 1

            except Exception as e:
                logging.info(i, e)

            if i % 1000 == 0:
                logging.info(f'\r i = {i}, # of template: {len(TemplateFreq)}, '
                             f'# of atom template: {len(templates_A)}, '
                             f'# of bond template: {len(templates_B)}')
        logging.info('\n total # of template: %s' % len(TemplateFreq))

        derived_templates = {'atom': templates_A, 'bond': templates_B}

        ofn = os.path.join(self.processed_data_path, "template_infos.csv")
        TemplateInfos = pd.DataFrame(
            {'Template': k,
             'edit_site': TemplateEdits[k],
             'change_H': TemplateHs[k],
             'change_C': TemplateCs[k],
             'change_S': TemplateSs[k],
             'Frequency': TemplateFreq[k]} for k in TemplateHs.keys())
        TemplateInfos.to_csv(ofn)

        for k, local_templates in derived_templates.items():
            ofn = os.path.join(self.processed_data_path, f"{k}_templates.csv")
            with open(ofn, "w") as of:
                writer = csv.writer(of)
                header = ['Template', 'Frequency', 'Class']
                writer.writerow(header)

                sorted_tuples = sorted(local_templates.items(), key=lambda item: item[1])
                for i, (template, template_freq) in enumerate(sorted_tuples):
                    writer.writerow([template, template_freq, i + 1])

    def match_templates(self):
        """Adapted from Run_preprocessing.py"""
        # load_templates()
        template_dicts = {}

        for site in ['atom', 'bond']:
            fn = os.path.join(self.processed_data_path, f"{site}_templates.csv")
            with open(fn, "r") as csv_file:
                csv_reader = csv.DictReader(csv_file)
                template_dict = {row["Template"].strip(): int(row["Class"]) for row in csv_reader}
                logging.info(f'loaded {len(template_dict)} {site} templates')
                template_dicts[site] = template_dict

        fn = os.path.join(self.processed_data_path, "template_infos.csv")
        with open(fn, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            template_infos = {
                row["template"]: {
                    "edit_site": eval(row["edit_site"]),
                    "frequency": int(row["Frequency"])
                } for row in csv_reader
            }
        logging.info('loaded total %s templates' % len(template_infos))

        # labeling_dataset()
        dfs = {}
        for phase, fn in [("train", self.train_file),
                          ("val", self.val_file),
                          ("test", self.test_file)]:
            with open(fn, "r") as csv_file:
                csv_reader = csv.DictReader(csv_file)
                rxns = [row["rxn_smiles"].strip() for row in csv_reader]
            reactants, products, reagents = [], [], []
            labels, frequency = [], []
            success = 0

            for i, rxn in enumerate(rxns):
                reactant, _, product = rxn.split(">")
                reagent = ''
                rxn_labels = []
                try:
                    rxn = {'reactants': reactant, 'products': product, '_id': i}
                    result = extract_from_reaction(rxn, self.setting)

                    template = result['reaction_smarts']
                    reactant = result['reactants']
                    product = result['products']
                    reagent = '.'.join(result['necessary_reagent'])
                    edits = {edit_type: edit_bond[0] for edit_type, edit_bond in result['edits'].items()}
                    H_change, Charge_change, Chiral_change = \
                        result['H_change'], result['Charge_change'], result['Chiral_change']
                    template_H = get_full_template(template, H_change, Charge_change, Chiral_change)

                    if template_H not in template_infos.keys():
                        reactants.append(reactant)
                        products.append(product)
                        reagents.append(reagent)
                        labels.append(rxn_labels)
                        frequency.append(0)
                        continue

                except Exception as e:
                    logging.info(i, e)
                    reactants.append(reactant)
                    products.append(product)
                    reagents.append(reagent)
                    labels.append(rxn_labels)
                    frequency.append(0)
                    continue

                edit_n = 0
                for edit_type in edits:
                    if edit_type == 'C':
                        edit_n += len(edits[edit_type]) / 2
                    else:
                        edit_n += len(edits[edit_type])

                if edit_n <= self.setting['max_edit_n']:
                    try:
                        success += 1
                        atom_sites, bond_sites = get_edit_site_retro(product)
                        for edit_type, edit in edits.items():
                            for e in edit:
                                if edit_type in ['A', 'R']:
                                    rxn_labels.append(
                                        ('a', atom_sites.index(e), template_dicts['atom'][template_H]))
                                else:
                                    rxn_labels.append(
                                        ('b', bond_sites.index(e), template_dicts['bond'][template_H]))
                        reactants.append(reactant)
                        products.append(product)
                        reagents.append(reagent)
                        labels.append(rxn_labels)
                        frequency.append(template_infos[template_H]['frequency'])

                    except Exception as e:
                        logging.info(i, e)
                        reactants.append(reactant)
                        products.append(product)
                        reagents.append(reagent)
                        labels.append(rxn_labels)
                        frequency.append(0)
                        continue

                    if i % 1000 == 0:
                        logging.info(f'\r Processing {self.data_name} {phase} data..., '
                                     f'success {success} data ({i}/{len(rxns)})')
                else:
                    logging.info(f'\nReaction # {i} has too many edits ({edit_n})...may be wrong mapping!')
                    reactants.append(reactant)
                    products.append(product)
                    reagents.append(reagent)
                    labels.append(rxn_labels)
                    frequency.append(0)

            logging.info(f'\nDerived templates cover {success / len(rxns): .3f} of {phase} data reactions')
            ofn = os.path.join(self.processed_data_path, f"preprocessed_{phase}.csv")
            dfs[phase] = pd.DataFrame(
                {'Reactants': reactants,
                 'Products': products,
                 'Reagents': reagents,
                 'Labels': labels,
                 'Frequency': frequency})
            dfs[phase].to_csv(ofn)

        # make_simulate_output()
        df = dfs["test"]
        ofn = os.path.join(self.processed_data_path, "simulate_output.txt")
        with open(ofn, 'w') as of:
            of.write('Test_id\tReactant\tProduct\t%s\n' % '\t'.join(
                [f'Edit {i + 1}\tProba {i + 1}' for i in range(self.setting['max_edit_n'])]))
            for i in df.index:
                labels = []
                for y in eval(df['Labels'][i]):
                    if y != 0:
                        labels.append(y)
                if not labels:
                    labels = [(0, 0)]
                string_labels = '\t'.join([f'{l}\t{1.0}' for l in labels])
                of.write('%s\t%s\t%s\t%s\n' % (i, df['Reactants'][i], df['Products'][i], string_labels))

        # combine_preprocessed_data()
        dfs["train"]['Split'] = ['train'] * len(dfs["train"])
        dfs["val"]['Split'] = ['val'] * len(dfs["val"])
        dfs["test"]['Split'] = ['test'] * len(dfs["test"])
        all_valid = dfs["train"].append(dfs["val"], ignore_index=True)
        all_valid = all_valid.append(dfs["test"], ignore_index=True)
        all_valid['Mask'] = [int(f >= self.setting['min_template_n']) for f in all_valid['Frequency']]
        ofn = os.path.join(self.processed_data_path, "labeled_data.csv")
        all_valid.to_csv(ofn, index=None)
        logging.info(f'Valid data size: {len(all_valid)}')
