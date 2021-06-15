def add_model_opts(parser):
    """Model options"""
    group = parser.add_argument_group("neuralsym_model")
    group.add("--min_freq", help="Minimum frequency of template in training data to be retained", type=int, default=1)
    group.add("--radius", help="Fingerprint radius", type=int, default=2)
    group.add("--fp_size", help="Fingerprint size", type=int, default=1000000)
    group.add("--final_fp_size", help="Fingerprint size", type=int, default=32681)


def add_train_opts(parser):
    """Training options"""
    group = parser.add_argument_group("retroxpert_train")
    group.add("--seed", help="random seed", type=int, default=123)
    group.add("--use_cpu", help="whether to use CPU", action="store_true")
    group.add("--load_checkpoint_s1", help="set to true to load trained S1 model", action="store_true")
    group.add("--batch_size", help="batch size", type=int, default=32)
    group.add("--epochs", help="no. of training epochs", type=int, default=80)
    group.add("--lr", help="learning rate", type=float, default=5e-4)
    group.add("--in_dim", help="dim of atom feature", type=int, default=47)
    # this was 47+657 originally, but semi-pattern count (657) is not deterministic
    group.add("--hidden_dim", help="hidden size", type=int, default=128)
    group.add("--heads", help="no. of attention heads", type=int, default=4)
    group.add("--gat_layers", help="no. of GAT layers", type=int, default=3)
