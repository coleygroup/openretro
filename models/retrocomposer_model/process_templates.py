import json
import pandas as pd
import multiprocessing
from rdkit import Chem
from tqdm import tqdm
from collections import Counter

from indigo import Indigo


# r = "Br-[CH2;+0:1].[OH;+0:2]"
# p = "[CH2;+0:1]-[O;H0;+0:2]"
# mr = Chem.MolFromSmarts(r)
# [atom.SetAtomMapNum(0) for atom in mr.GetAtoms()]
# r = Chem.MolToSmarts(mr)
# print(r, len(r.split('.')))
# reacts = r.split('.')
#
# mp = Chem.MolFromSmarts(p)
# [atom.SetAtomMapNum(0) for atom in mp.GetAtoms()]
# p = Chem.MolToSmarts(mp)
# print(p, len(p.split('.')))
# prods = p.split('.')
#
# rxn = Indigo().loadReactionSmarts("Br-[CH2;+0:1].[OH;+0:2]>>C-[O;H0;D2;+0:1]-[c:2]")
# res = rxn.automap("discard")
#
#
# print("mol number:", rxn.countMolecules())
# num_react = len(r.split('.'))
# num_prod = len(p.split('.'))
# assert rxn.countMolecules() == num_react + num_prod
# react_mapnums = []
# prod_mapnums = []
# for idx, mol in enumerate(rxn.iterateMolecules()):
#     # if idx < num_react:
#     #     assert Chem.MolFromSmarts(reacts[idx]).GetNumAtoms() == mol.countAtoms()
#     # else:
#     #     assert Chem.MolFromSmarts(prods[idx - num_react]).GetNumAtoms() == mol.countAtoms()
#
#     print('mol atom number:', mol.countAtoms())
#     print(mol.smiles(), mol.smarts())
#     for atom in mol.iterateAtoms():
#         # print("Atom ", atom.index(), atom.atomicNumber(), atom.symbol())
#         mapnum = rxn.atomMappingNumber(atom)
#         if idx < num_react:
#             react_mapnums.append(mapnum)
#         else:
#             prod_mapnums.append(mapnum)
#
# # assert len(mr.GetAtoms()) == len(react_mapnums)
# for atom, num in zip(mr.GetAtoms(), react_mapnums):
#     atom.SetAtomMapNum(num)
#
# r = Chem.MolToSmarts(mr)
#
# # assert len(mp.GetAtoms()) == len(prod_mapnums)
# for atom, num in zip(mp.GetAtoms(), prod_mapnums):
#     atom.SetAtomMapNum(num)
#
# p = Chem.MolToSmarts(mp)
#
# print('after mapping:', r, p)
#
# exit(1)

def cano_smiles(smiles):
    smis = smiles.split('.')
    smis = [cano_smiles_(smi) for smi in smis]
    if None in smis:
        return None
    return cano_smiles_('.'.join(smis))

def cano_smiles_(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]
        return Chem.MolToSmiles(mol)
    else:
        return None

def cano_smarts(smarts):
    smts = smarts.split('.')
    smts = [cano_smarts_(smt) for smt in smts]
    if None in smts:
        return None
    return cano_smarts_('.'.join(smts))

def cano_smarts_(smarts):
    tmp = Chem.MolFromSmarts(smarts)
    if tmp is None:
        return None
    [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
    cano = Chem.MolToSmarts(tmp)
    if '[[se]]' in cano:  # strange parse error
        cano = smarts
    return cano


def compose_tmpl(prod, react):
    """
        input product and reactant smarts
        return mapped reaction smarts if possible, other return False
    """
    rxn = Indigo().loadReactionSmarts(react + ">>" + prod)
    res = rxn.automap("discard")
    if not res: return False

    reacts = react.split('.')
    prods = prod.split('.')
    num_react = len(reacts)
    num_prod = len(prods)
    if rxn.countMolecules() != num_react + num_prod:
        print('rxn.countMolecules() != num_react + num_prod', rxn.countMolecules(), num_react, num_prod)
        return False

    mapnums = []
    react_mapnum_len = 0
    for idx, mol in enumerate(rxn.iterateMolecules()):
        mapnums.extend([rxn.atomMappingNumber(atom) for atom in mol.iterateAtoms()])
        if idx < num_react:
            react_mapnum_len = len(mapnums)
        #     assert Chem.MolFromSmarts(reacts[idx]).GetNumAtoms() == mol.countAtoms()
        # else:
        #     assert Chem.MolFromSmarts(prods[idx - num_react]).GetNumAtoms() == mol.countAtoms()

    mr = Chem.MolFromSmarts(react)
    # assert len(mr.GetAtoms()) == react_mapnum_len
    for atom, num in zip(mr.GetAtoms(), mapnums[:react_mapnum_len]):
        atom.SetAtomMapNum(num)
    r = Chem.MolToSmarts(mr)

    mp = Chem.MolFromSmarts(prod)
    # assert len(mp.GetAtoms()) == len(mapnums) - react_mapnum_len
    for atom, num in zip(mp.GetAtoms(), mapnums[react_mapnum_len:]):
        atom.SetAtomMapNum(num)
    p = Chem.MolToSmarts(mp)
    return p + ">>" + r


def process_tmpl(task):
    template, template_general = task
    prod_cano_general, reacts_cano_general = None, None,
    if template_general:
        prod, reacts= template_general.split(">>")
        prod_cano_general = cano_smarts(prod)
        reacts_cano_general = [cano_smarts(r) for r in reacts.split('.')]

    prod_cano, reacts_cano = None, None
    if template:
        prod, reacts = template.split(">>")
        prod_cano = cano_smarts(prod)
        reacts_cano = [cano_smarts(r) for r in reacts.split('.')]
        # compose template by adding mapping numbers
        # template_composed = compose_tmpl(prod_cano, '.'.join(reacts_cano))
        # print('\ntemplate:', template)
        # print('template_composed:', template_composed)
    return prod_cano_general, reacts_cano_general, prod_cano, reacts_cano


if __name__ == "__main__":
    # r1 = 'C[C@@H]1OCCN[C@@H]1C.FC(F)c1nc2ccccc2n1-c1nc(Cl)cc(Cl)n1'
    # r2 = 'C[C@@H]1NCCO[C@@H]1C.FC(F)c1nc2ccccc2n1-c1nc(Cl)cc(Cl)n1'
    # m1 = Chem.MolFromSmiles(r1)
    # m2 = Chem.MolFromSmiles(r2)
    # print(m1.HasSubstructMatch(m2, useChirality=True))
    # print(m2.HasSubstructMatch(m1, useChirality=True))
    # print(Chem.MolToSmiles(m1), Chem.MolToSmiles(m1))
    # exit(1)


    # tmpl_unique = {}
    # tmpl_cnt = [Counter(), Counter()]
    # for dataset in ['test', 'train', 'valid']:
    #     data_file = 'data/USPTO50K/templates_{}.json'.format(dataset)
    #     templates = json.load(open(data_file))
    #     tasks = []
    #     for key, val in templates.items():
    #         tasks.append([val['reaction_smarts'], val['reaction_smarts_general']])
    #
    #     tmpl_general_counter = {
    #         'product': Counter(),
    #         'reactant': Counter(),
    #     }
    #     tmpl_counter = {
    #         'product': Counter(),
    #         'reactant': Counter(),
    #     }
    #
    #     with multiprocessing.Pool(8) as pool:
    #         for res in tqdm(pool.imap_unordered(process_tmpl, tasks), total=len(tasks)):
    #             prod_cano_general, reacts_cano_general, prod_cano, reacts_cano = res
    #             if prod_cano_general:
    #                 tmpl_general_counter['product'][prod_cano_general] += 1
    #             if reacts_cano_general is not None and len(reacts_cano_general):
    #                 tmpl_general_counter['reactant'].update(reacts_cano_general)
    #                 if prod_cano_general:
    #                     react = '.'.join(reacts_cano_general)
    #                     tmpl = react + '>>' + prod_cano_general
    #                     tmpl_cnt[0][tmpl] += 1
    #
    #             if prod_cano:
    #                 tmpl_counter['product'][prod_cano] += 1
    #             if reacts_cano is not None and len(reacts_cano):
    #                 tmpl_counter['reactant'].update(reacts_cano)
    #                 if prod_cano:
    #                     react = '.'.join(reacts_cano)
    #                     tmpl = react + '>>' + prod_cano
    #                     tmpl_cnt[1][tmpl] += 1
    #
    #     tmpl_unique[dataset] = {
    #         'tmpl': tmpl_counter,
    #         'tmpl_general': tmpl_general_counter
    #     }
    #
    # print('tmpl_cnt:', len(tmpl_cnt[0]), len(tmpl_cnt[1]))
    # with open('data/USPTO50K/templates_comps.json', 'w') as f:
    #     json.dump(tmpl_unique, f, indent=4)


    # find cano product smart mapping to templates
    cano_prod_to_templates = {}
    data_file = 'data/USPTO50K/templates_train.json'
    templates = json.load(open(data_file))
    for idx, val in templates.items():
        items = val['reaction_smarts'].split('>>')
        if len(items) < 2:
            continue
        cano_prod = cano_smarts(items[0])
        if cano_prod not in cano_prod_to_templates:
            cano_prod_to_templates[cano_prod] = []
        cano_prod_to_templates[cano_prod].append(val['reaction_smarts'])

    with open('data/USPTO50K/templates_comps.json') as f:
        tmpl = json.load(f)
    general = 'tmpl'
    products = tmpl['train'][general]['product']
    products_filtered = dict(filter(lambda elem: elem[1] > 0, products.items()))
    reactants = tmpl['train'][general]['reactant']
    reactants_filtered = dict(filter(lambda elem: elem[1] > 0, reactants.items()))

    print('products and reactants size:', len(products), len(reactants), )
    print('products_filtered and reactants_filtered size:', len(products_filtered), len(reactants_filtered), )


    # check template coverage
    for dataset in ['test', 'valid']:
        prod = tmpl[dataset][general]['product']
        p_cnt, p_total = 0, 0
        for p, cnt in prod.items():
            if p in products_filtered:
                p_cnt += cnt
            p_total += cnt
        print(dataset, 'testing products coverage: ', p_cnt, p_total, p_cnt / p_total)

        react = tmpl[dataset][general]['reactant']
        p_cnt, p_total = 0, 0
        for p, cnt in react.items():
            if p in reactants_filtered:
                p_cnt += cnt
            p_total += cnt
        print(dataset, 'testing reactant coverage: ', p_cnt, p_total, p_cnt / p_total)




