from rdkit import Chem


def canonicalize_smiles(smiles: str, remove_atom_number: bool = True):
    """Adapted from Molecular Transformer"""
    smiles = "".join(smiles.split())

    mol = Chem.MolFromSmiles(smiles)
    if remove_atom_number:
        [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]

    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    else:
        return ""
