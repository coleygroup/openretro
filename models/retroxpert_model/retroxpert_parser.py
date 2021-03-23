def add_model_opts(parser):
    """Model options"""
    group = parser.add_argument_group("retroxpert_model")
    group.add("--typed", help="whether reaction class is given", action="store_true")


def add_preprocess_opts(parser):
    """Preprocessing options"""
    group = parser.add_argument_group("retroxpert_preprocess")
    group.add("--min_freq", help="minimum frequency for patterns to be kept", type=int, default=2)
    group.add("--model_path_s1", help="model path from stage 1 (needed for stage 2)", type=str, default="")
    group.add("--load_checkpoint_s1", help="set to true to load trained S1 model", action="store_true")


def add_train_opts(parser):
    """Training options"""
    group = parser.add_argument_group("retroxpert_train")
    group.add("--seed", help="random seed", type=int, default=123)
    group.add("--use_cpu", help="whether to use CPU", action="store_true")
    group.add("--load", help="whether to load from checkpoint", action="store_true")
    group.add("--batch_size", help="batch size", type=int, default=32)
    group.add("--epochs", help="no. of training epochs", type=int, default=80)
    group.add("--lr", help="learning rate", type=float, default=5e-4)
    group.add("--in_dim", help="dim of atom feature", type=int, default=47+657)
    group.add("--hidden_dim", help="hidden size", type=int, default=128)
    group.add("--heads", help="no. of attention heads", type=int, default=4)
    group.add("--gat_layers", help="no. of GAT layers", type=int, default=3)
