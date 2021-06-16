def add_model_opts(parser):
    """Model options"""
    group = parser.add_argument_group("neuralsym_model")
    group.add("--min_freq", help="Minimum frequency of template in training data to be retained", type=int, default=1)
    group.add("--radius", help="Fingerprint radius", type=int, default=2)
    group.add("--fp_size", help="Fingerprint size", type=int, default=1000000)
    group.add("--final_fp_size", help="Fingerprint size", type=int, default=32681)


def add_train_opts(parser):
    """Training options"""
    group = parser.add_argument_group("neuralsym_train")
    group.add("--model_arch", help="['Highway', 'FC']", type=str, default='Highway')
    # training params
    group.add("--seed", help="random seed", type=int, default=0)
    group.add("--bs", help="batch size", type=int, default=128)
    group.add("--bs_eval", help="batch size (valid/test)", type=int, default=256)
    group.add("--learning_rate", help="learning rate", type=float, default=1e-3)
    group.add("--epochs", help="num. of epochs", type=int, default=30)
    group.add("--early_stop", help="whether to use early stopping", action="store_true")
    group.add("--early_stop_patience",
              help="num. of epochs tolerated without improvement in criteria before early stop",
              type=int, default=2)
    group.add("--early_stop_min_delta",
              help="min. improvement in criteria needed to not early stop", type=float, default=1e-4)
    group.add("--lr_scheduler_factor",
              help="factor by which to reduce LR (ReduceLROnPlateau)", type=float, default=0.3)
    group.add("--lr_scheduler_patience",
              help="num. of epochs with no improvement after which to reduce LR (ReduceLROnPlateau)",
              type=int, default=1)
    group.add("--lr_cooldown", help="epochs to wait before resuming normal operation (ReduceLROnPlateau)",
              type=int, default=0)
    # model params
    group.add("--hidden_size", help="hidden size", type=int, default=512)
    group.add("--depth", help="depth", type=int, default=5)
