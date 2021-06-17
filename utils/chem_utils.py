from rdkit import Chem


def canonicalize_smiles(smiles: str, remove_atom_number: bool = True):
    """Adapted from Molecular Transformer"""
    smiles = "".join(smiles.split())
    cano_smiles = ""

    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        if remove_atom_number:
            [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]

        cano_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        # Sometimes stereochem takes another canonicalization... (just in case)
        mol = Chem.MolFromSmiles(cano_smiles)
        if mol is not None:
            cano_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

    return cano_smiles
