
import re
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import AllChem


#  Get the dict maps each atom index to the mapping number.
def get_atomidx2mapnum(mol):
    atomidx2mapnum = {}
    for atom in mol.GetAtoms():
        assert atom.GetAtomMapNum() > 0
        atomidx2mapnum[atom.GetIdx()] = atom.GetAtomMapNum()
    return atomidx2mapnum

#  Get the dict maps each mapping number to the atom index .
def get_mapnum2atomidx(mol):
    mapnum2atomidx = {}
    for atom in mol.GetAtoms():
        assert atom.GetAtomMapNum() > 0
        mapnum2atomidx[atom.GetAtomMapNum()] = atom.GetIdx()
    return mapnum2atomidx

def get_mapnum(smiles):
    item = re.findall('(?<=:)\d+', smiles)
    item = list(map(int, item))
    return item

def cano_smiles(smiles, remove_mapping=True):
    smis = smiles.split('.')
    smis = [cano_smiles_(smi, remove_mapping) for smi in smis]
    if None in smis:
        return None
    return cano_smiles_('.'.join(smis), remove_mapping)

def cano_smiles_(smiles, remove_mapping=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        if remove_mapping:
            [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]
        return Chem.MolToSmiles(mol)
    else:
        return None

def get_pattern_fingerprint_bitstr(mol, fpSize=1024):
    return Chem.PatternFingerprint(mol, fpSize=fpSize).ToBitString()

def get_pattern_fingerprint_onbits(mol, fpSize=1024):
    return set(Chem.PatternFingerprint(mol, fpSize=fpSize).GetOnBits())

def get_morgan_fingerprint(mol, nBits=1024, useFeatures=True):
    return AllChem.GetMorganFingerprint(mol, radius=2, nBits=nBits, useFeatures=useFeatures)

def smarts_to_cano_smiles(smarts):
    mol = Chem.MolFromSmarts(smarts)
    [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]
    return Chem.MolToSmiles(mol, allBondsExplicit=True, allHsExplicit=False)

def cano_smarts(smarts, remove_mapping=True):
    smts = smarts.split('.')
    smts = [cano_smarts_(smt, remove_mapping) for smt in smts]
    if None in smts:
        return None
    return cano_smarts_('.'.join(smts), remove_mapping)

def cano_smarts_(smarts, remove_mapping=True):
    tmp = Chem.MolFromSmarts(smarts)
    if tmp is None:
        return None
    if remove_mapping:
        [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
    cano = Chem.MolToSmarts(tmp)
    if '[[se]]' in cano:  # strange parse error
        cano = smarts
    return cano

def canonicalize_rxn(reactant, product):
    # canonicalize product SMILES by re-assigning atom mapping numbers
    # according to the canonical atom order.
    mol = Chem.MolFromSmiles(product)
    index2mapnums = get_atomidx2mapnum(mol)
    # canonicalize the product smiles
    mol_cano = Chem.RWMol(mol)
    [atom.SetAtomMapNum(0) for atom in mol_cano.GetAtoms()]
    mol_cano = Chem.MolFromSmiles(Chem.MolToSmiles(mol_cano))
    matches = mol.GetSubstructMatches(mol_cano)
    if matches:
        mapnums_old2new = {}
        for atom, mat in zip(mol_cano.GetAtoms(), matches[0]):
            mapnums_old2new[index2mapnums[mat]] = 1 + atom.GetIdx()
            # update product mapping numbers according to canonical atom order
            # to completely remove the potential information leak
            atom.SetAtomMapNum(1 + atom.GetIdx())
        product = Chem.MolToSmiles(mol_cano)
        # update reactant mapping numbers accordingly
        mol_react = Chem.MolFromSmiles(reactant)
        for atom in mol_react.GetAtoms():
            if atom.GetAtomMapNum() > 0:
                atom.SetAtomMapNum(mapnums_old2new[atom.GetAtomMapNum()])
        reactant = Chem.MolToSmiles(mol_react)
    else:
        print('canonicalization fail:', reactant, product)

    return reactant, product

def assign_mapping_number(smi):
    # assign mapping number according to canonical atom index
    mol_cano = Chem.RWMol(Chem.MolFromSmiles(smi))
    [atom.SetAtomMapNum(0) for atom in mol_cano.GetAtoms()]
    mol_cano = Chem.MolFromSmiles(Chem.MolToSmiles(mol_cano))
    [atom.SetAtomMapNum(atom.GetIdx() + 1) for atom in mol_cano.GetAtoms()]
    return Chem.MolToSmiles(mol_cano)

