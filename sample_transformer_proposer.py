from models.transformer_model.transformer_proposer import TransformerProposer


def get_basic_config():
    basic_config = {
        "model_name": "model_name",
        "processed_data_path": "./data/transformer-karpov/processed/",
        "model_path": "./checkpoints/transformer-karpov/model_step_500000.pt",
        "n_best": 10,
        "beam_size": 20
    }

    return basic_config


def transformer_proposer_test():
    model_config = get_basic_config()

    proposer = TransformerProposer(
        model_name=model_config["model_name"],
        model_args=None,
        model_config=model_config,
        data_name="",
        processed_data_path=model_config["processed_data_path"],
        model_path=model_config["model_path"]
    )

    product_smiles = [
        "CC(C)(C)OC(=O)NCCc1ccc(N)cc1F",
        "COC(=O)CCC(=O)c1ccc(OC2CCCCO2)cc1O"
    ]

    results = proposer.propose(product_smiles)
    print(results)
    """
    List of n[{"reactants": list of topk reactants,
               "scores": list of topk scores}]
    """


if __name__ == "__main__":
    transformer_proposer_test()
