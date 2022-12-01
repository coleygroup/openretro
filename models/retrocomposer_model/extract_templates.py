import argparse
import json
import numpy as np
import os
import logging
import multiprocessing
import pandas as pd

from collections import Counter
from rdkit import Chem
from tqdm import tqdm

from rdchiral.template_extractor import extract_from_reaction
from rdchiral.main import rdchiralReaction, rdchiralReactants, rdchiralRun
from models.retrocomposer_model.chemutils import canonicalize_rxn
from models.retrocomposer_model.chemutils import cano_smiles, cano_smiles_, cano_smarts, cano_smarts_
from models.retrocomposer_model.chemutils import smarts_to_cano_smiles, get_pattern_fingerprint_bitstr


class _Reactor(object):
    def __init__(self):
        self.rxn_cooked = {}
        self.src_cooked = {}
        self.cached_results = {}

    def get_rxn(self, rxn):
        p, a, r = rxn.split('>')
        if '.' in p:  # we assume the product has only one molecule
            if p[0] != '(':
                p = '('+p+')'
        rxn = '>'.join((p, a, r))
        if not rxn in self.rxn_cooked:
            try:
                t = rdchiralReaction(rxn)
            except:
                t = None
            self.rxn_cooked[rxn] = t
        return self.rxn_cooked[rxn]

    def get_src(self, smiles):
        if not smiles in self.src_cooked:
            self.src_cooked[smiles] = rdchiralReactants(smiles)
        return self.src_cooked[smiles]

    def run_reaction(self, src, template, keep_mapnums=False):
        key = (src, template)
        if key in self.cached_results:
            return self.cached_results[key]
        rxn = self.get_rxn(template)
        src = self.get_src(src)
        if rxn is None or src is None:
            return None
        try:
            outcomes = rdchiralRun(rxn, src, keep_mapnums=keep_mapnums)
            self.cached_results[key] = outcomes
        except:
            self.cached_results[key] = None
        return self.cached_results[key]

Reactor = _Reactor()


def get_tpl(task):
    idx, react, prod, id, cls = task
    # reassign product mapping numbers
    react, prod = canonicalize_rxn(react, prod)
    reaction = {'_id': idx, 'reactants': react, 'products': prod}
    template = extract_from_reaction(reaction)
    pred_reacts = []
    retro_okay_list = []
    reaction_smarts = ''
    cano_react = cano_smiles(react)
    if 'reaction_smarts' in template:
        reaction_smarts = template['reaction_smarts']
        pred_mols = Reactor.run_reaction(prod, reaction_smarts)
        if pred_mols and len(pred_mols):
            react_mol = Chem.MolFromSmiles(cano_react)
            for pred_react in pred_mols:
                pred_react = cano_smiles(pred_react)
                retro_okay = False
                if cano_react == pred_react:
                    retro_okay = 'exact_match'
                else:
                    pred_react_mol = Chem.MolFromSmiles(pred_react)
                    if react_mol.HasSubstructMatch(pred_react_mol, useChirality=True) and \
                            pred_react_mol.HasSubstructMatch(react_mol, useChirality=True):
                        retro_okay = 'equal_mol'
                pred_reacts.append(pred_react)
                retro_okay_list.append(retro_okay)

    return idx, react, prod, reaction_smarts, cano_react, retro_okay_list, id, cls


def match_template(task):
    idx, val = task
    reactant = cano_smiles(val['reactant'])
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol_prod = Chem.MolFromSmiles(val['product'], params)
    prod_fp_vec = int(get_pattern_fingerprint_bitstr(mol_prod), 2)

    sequences = []
    template_cands = []
    templates_list = []
    atom_indexes_fp_labels = {}
    # multiple templates may be valid for a reaction, find all of them
    for prod_smarts_fp_idx, prod_smarts_tmpls in prod_smarts_fp_to_templates.items():
        prod_smarts_fp_idx = int(prod_smarts_fp_idx)
        prod_smarts_fp = prod_smarts_fp_list[prod_smarts_fp_idx]
        for prod_smarts_idx, tmpls in prod_smarts_tmpls.items():
            # skip if fingerprint not match
            if (prod_smarts_fp & prod_fp_vec) < prod_smarts_fp:
                continue
            prod_smarts_idx = int(prod_smarts_idx)
            prod_smarts = prod_smarts_list[prod_smarts_idx]
            if prod_smarts not in smarts_mol_cache:
                smarts_mol_cache[prod_smarts] = Chem.MergeQueryHs(Chem.MolFromSmarts(prod_smarts))
            # we need also find matched atom indexes
            matches = mol_prod.GetSubstructMatches(smarts_mol_cache[prod_smarts])
            if len(matches):
                found_okay_tmpl = False
                for tmpl in tmpls:
                    pred_mols = Reactor.run_reaction(val['product'], tmpl)
                    if reactant and pred_mols and (reactant in pred_mols):
                        found_okay_tmpl = True
                        template_cands.append(templates_train.index(tmpl))
                        templates_list.append(tmpl)
                        reacts = tmpl.split('>>')[1].split('.')
                        if len(reacts) > 2: logging.info(f'too many reacts: {reacts}, {id}')
                        seq_reacts = [react_smarts_list.index(cano_smarts(r)) for r in reacts]
                        seq = [prod_smarts_fp_idx] + sorted(seq_reacts)
                        sequences.append(seq)
                # for each prod center, there may be multiple matches
                for match in matches:
                    match = tuple(sorted(match))
                    if match not in atom_indexes_fp_labels:
                        atom_indexes_fp_labels[match] = {}
                    if prod_smarts_fp_idx not in atom_indexes_fp_labels[match]:
                        atom_indexes_fp_labels[match][prod_smarts_fp_idx] = [[], []]
                    atom_indexes_fp_labels[match][prod_smarts_fp_idx][0].append(prod_smarts_idx)
                    atom_indexes_fp_labels[match][prod_smarts_fp_idx][1].append(found_okay_tmpl)

    reaction_center_cands = []
    reaction_center_cands_labels = []
    reaction_center_cands_smarts = []
    reaction_center_atom_indexes = []
    for atom_index in sorted(atom_indexes_fp_labels.keys()):
        for fp_idx, val in atom_indexes_fp_labels[atom_index].items():
            reaction_center_cands.append(fp_idx)
            reaction_center_cands_smarts.append(val[0])
            reaction_center_cands_labels.append(True in val[1])
            reaction_center_atom_indexes.append(atom_index)

    tmpl_res = {
        'templates': templates_list,
        'template_cands': template_cands,
        'template_sequences': sequences,
        'reaction_center_cands': reaction_center_cands,
        'reaction_center_cands_labels': reaction_center_cands_labels,
        'reaction_center_cands_smarts': reaction_center_cands_smarts,
        'reaction_center_atom_indexes': reaction_center_atom_indexes,
    }

    return idx, tmpl_res


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='USPTO50K', help='dataset: USPTO50K')
    parser.add_argument('--prod_k', type=int, default='1', help='product min counter to be kept')
    parser.add_argument('--react_k', type=int, default='1', help='reactant min counter to be kept')

    args = parser.parse_args()
    print('extract templates for dataset {}...'.format(args.dataset))
    assert args.dataset in ['USPTO50K']


    for data_set in ['valid', 'train', 'test']:
        data_file = './data/{}/raw_{}.csv'.format(args.dataset, data_set)
        csv = pd.read_csv(data_file)
        data_file = './data/{}/templates_{}.json'.format(args.dataset, data_set)
        if not os.path.isfile(data_file):
            reaction_list = csv['reactants>reagents>production']
            reactant_list = list(map(lambda x: x.split('>')[0], reaction_list))
            product_list = list(map(lambda x: x.split('>')[-1], reaction_list))
            ids = csv['id'].tolist()
            classes = csv['class'].tolist()
            rxns = []
            for idx, r in enumerate(reactant_list):
                rxns.append((idx, r, product_list[idx], ids[idx], classes[idx]))
            print('total rxns:', len(rxns))

            cnt = 0
            train_templates = {}
            pool = multiprocessing.Pool(16)
            for result in tqdm(pool.imap_unordered(get_tpl, rxns), total=len(rxns)):
                idx, react, prod, reaction_smarts, cano_reacts, retro_okay, id, cls = result
                cnt += 'exact_match' in retro_okay or 'equal_mol' in retro_okay
                train_templates[idx] = {
                    'id': id,
                    'class': cls,
                    'reactant': react,
                    'product': prod,
                    'reaction_smarts': reaction_smarts,
                    'cano_reactants': cano_reacts,
                }

            print('retro_okay cnt:', cnt, len(rxns), cnt / len(rxns))
            with open(data_file, 'w') as f:
                json.dump(train_templates, f, indent=4)

    # find cano templates
    templates_cano_train = './data/{}/templates_cano_train.json'.format(args.dataset)
    if False and os.path.isfile(templates_cano_train):
        data = json.load(open(templates_cano_train))
        templates_train = data['templates_train']
        react_smarts_list = data['react_smarts_list']
        prod_smarts_list = data['prod_smarts_list']
        prod_smarts_fp_list = data['prod_smarts_fp_list']
        fp_prod_smarts_dict = data['fp_prod_smarts_dict']
        prod_smarts_fp_to_templates = data['prod_smarts_fp_to_templates']
        for key, val in data.items():
            print(key, len(val))
    else:
        print('find all cano training templates')
        data_file = './data/{}/templates_{}.json'.format(args.dataset, 'train')
        templates_data_train = json.load(open(data_file))
        smarts_fp_cache = {}
        fp_prod_smarts_dict = {}
        cano_react_smarts_dict = Counter()
        for idx, val in tqdm(templates_data_train.items()):
            items = val['reaction_smarts'].split('>>')
            if len(items) == 2 and items[0] and items[1]:
                prod_smarts, reacts_smarts = items
                prod_smarts = cano_smarts_(prod_smarts)
                if prod_smarts not in smarts_fp_cache:
                    mol = Chem.MergeQueryHs(Chem.MolFromSmarts(prod_smarts))
                    smarts_fp_cache[prod_smarts] = int(get_pattern_fingerprint_bitstr(mol), 2)
                if smarts_fp_cache[prod_smarts] not in fp_prod_smarts_dict:
                    fp_prod_smarts_dict[smarts_fp_cache[prod_smarts]] = {'cnt': 0, 'cano_smarts': []}
                fp_prod_smarts_dict[smarts_fp_cache[prod_smarts]]['cnt'] += 1
                fp_prod_smarts_dict[smarts_fp_cache[prod_smarts]]['cano_smarts'].append(prod_smarts)

                cano_reacts_smarts = cano_smarts(reacts_smarts)
                cano_react_smarts_dict.update(cano_reacts_smarts.split('.'))
                val['cano_reaction_smarts'] = prod_smarts + '>>' + cano_reacts_smarts
            else:
                print('invalid reaction_smarts:', idx, items)

        print('fp_prod_smarts_dict and cano_react_smarts_dict size:', len(fp_prod_smarts_dict), len(cano_react_smarts_dict))
        print('smarts filter threshold: ', args.prod_k, args.react_k)

        prod_smarts_list = set()
        prod_smarts_fp_to_remove = []
        # filter product smarts less then the frequency
        for fp, val in fp_prod_smarts_dict.items():
            if val['cnt'] < args.prod_k:
                prod_smarts_fp_to_remove.append(fp)
            else:
                fp_prod_smarts_dict[fp] = list(set(val['cano_smarts']))
                prod_smarts_list.update(fp_prod_smarts_dict[fp])

        [fp_prod_smarts_dict.pop(fp) for fp in prod_smarts_fp_to_remove]
        prod_smarts_fp_list = sorted(fp_prod_smarts_dict.keys())
        prod_smarts_list = sorted(prod_smarts_list)
        # find smarts indexes
        for fp, val in fp_prod_smarts_dict.items():
            fp_prod_smarts_dict[fp] = [prod_smarts_list.index(v) for v in val]

        cano_react_smarts_dict = dict(filter(lambda elem: elem[1] >= args.react_k, cano_react_smarts_dict.items()))
        # sort reactants by frequency from high to low
        react_smarts_list = [k for k, v in
                             sorted(cano_react_smarts_dict.items(), key=lambda item: item[1], reverse=True)]

        print('after filtering, prod_smarts_fp_list and cano_react_smarts_list size:', len(prod_smarts_fp_list),
              len(cano_react_smarts_dict))

        prod_smarts_fp_to_templates = {}
        for idx, val in tqdm(templates_data_train.items()):
            if 'cano_reaction_smarts' in val:
                cano_prod_smarts, cano_reacts_smarts = val['cano_reaction_smarts'].split('>>')
                if cano_prod_smarts not in smarts_fp_cache:
                    mol = Chem.MergeQueryHs(Chem.MolFromSmarts(cano_prod_smarts))
                    smarts_fp_cache[cano_prod_smarts] = int(get_pattern_fingerprint_bitstr(mol), 2)
                if smarts_fp_cache[cano_prod_smarts] not in fp_prod_smarts_dict:
                    print('skip cano_prod_smarts:', idx, cano_prod_smarts)
                    continue

                cano_reacts_smarts = set(cano_reacts_smarts.split('.'))
                if not cano_reacts_smarts.issubset(cano_react_smarts_dict):
                    print('skip cano_reacts_smarts:', idx, cano_reacts_smarts)
                    continue

                cano_prod_smarts_fp_idx = prod_smarts_fp_list.index(smarts_fp_cache[cano_prod_smarts])
                if cano_prod_smarts_fp_idx not in prod_smarts_fp_to_templates:
                    prod_smarts_fp_to_templates[cano_prod_smarts_fp_idx] = {}
                cano_prod_smarts_idx = prod_smarts_list.index(cano_prod_smarts)
                if cano_prod_smarts_idx not in prod_smarts_fp_to_templates[cano_prod_smarts_fp_idx]:
                    prod_smarts_fp_to_templates[cano_prod_smarts_fp_idx][cano_prod_smarts_idx] = set()
                prod_smarts_fp_to_templates[cano_prod_smarts_fp_idx][cano_prod_smarts_idx].add(val['reaction_smarts'])

        tmpl_lens = []
        templates_train = set()
        for fp, val in prod_smarts_fp_to_templates.items():
            for cano_prod_smarts, tmpls in val.items():
                tmpl_lens.append(len(tmpls))
                templates_train.update(tmpls)
                prod_smarts_fp_to_templates[fp][cano_prod_smarts] = list(tmpls)
        print('#average template variants per cano_prod_smarts:', np.mean(tmpl_lens))

        print('templates_data_train:', len(templates_data_train))
        templates_train = sorted(list(templates_train))
        data = {
            'templates_train': templates_train,
            'react_smarts_list': react_smarts_list,
            'prod_smarts_list': prod_smarts_list,
            'prod_smarts_fp_list': prod_smarts_fp_list,
            'fp_prod_smarts_dict': fp_prod_smarts_dict,
            'prod_smarts_fp_to_templates': prod_smarts_fp_to_templates,
        }
        for key, val in data.items():
            print(key, len(val))
        with open(templates_cano_train, 'w') as f:
            json.dump(data, f, indent=4)

    smarts_mol_cache = {}
    smarts_fp_cache = {}
    # find all applicable templates for each reaction
    # since multiple templates may be valid for a reaction
    for data_set in ['test', 'train', 'valid']:
        data_file = './data/{}/templates_{}.json'.format(args.dataset, data_set)
        data_file_new = './data/{}/templates_{}_new.json'.format(args.dataset, data_set)
        print('find all applicable templates for each reaction:', data_file)
        rxn_templates = {}
        templates_data = json.load(open(data_file))
        tasks = [(idx, rxn) for idx, rxn in templates_data.items()]
        # for task in tasks: match_template(task)
        cnt = 0
        with multiprocessing.Pool(16) as pool:
            for res in tqdm(pool.imap_unordered(match_template, tasks), total=len(tasks)):
                idx, tmpl_res = res
                templates_data[idx].update(tmpl_res)
                cnt += len(tmpl_res['templates']) > 0
        print('template coverage: {} for {} dataset'.format(cnt / len(tasks), data_set))
        with open(data_file, 'w') as f:
            json.dump(templates_data, f, indent=4)
