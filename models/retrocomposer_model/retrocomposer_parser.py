def add_model_opts(parser):
    """Model options"""
    group = parser.add_argument_group("retroxpert_model")
    group.add('--num_layer', type=int, default=6,
              help='number of GNN message passing layers (default: 6).')
    group.add('--emb_dim', type=int, default=300,
              help='embedding dimensions (default: 300)')
    group.add('--graph_pooling', type=str, default="attention",
              help='graph level pooling (sum, mean, max, set2set, attention)')
    group.add('--JK', type=str, default="concat",
              help='how the node features across layers are combined. last, sum, max or concat')
    group.add('--atom_feat_dim', type=int, default=45,
              help="atom feature dimension.")
    group.add('--bond_feat_dim', type=int, default=12,
              help="bond feature dimension.")
    group.add('--onehot_center', action='store_true', default=False,
              help='reaction center encoding: onehot or subgraph')
    group.add('--center_loss_type', type=str, default='ce',
              help='loss type (bce or ce) for reaction center prediction')
    group.add('--dropout_ratio', type=float, default=0.2,
              help='dropout ratio (default: 0.5)')


def add_preprocess_opts(parser):
    """Preprocessing options"""
    group = parser.add_argument_group("retrocomposer_preprocess")
    group.add('--prod_k', type=int, default='1', help='product min counter to be kept')
    group.add('--react_k', type=int, default='1', help='reactant min counter to be kept')


def add_train_opts(parser):
    """Training options"""
    group = parser.add_argument_group("retroxpert_train")
    group.add("--seed", help="random seed", type=int, default=0)
    group.add("--use_cpu", help="whether to use CPU", action="store_true")
    group.add("--batch_size", help="batch size", type=int, default=32)
    group.add("--epochs", help="no. of training epochs", type=int, default=80)
    group.add("--lr", type=float, default=0.0003, help='learning rate (default: 0.0003)')
    group.add('--decay', type=float, default=0, help='weight decay (default: 0)')
    group.add('--multiprocess', action='store_true', help='train a model with multi process')
    group.add('--num_process', type=int, default=4, help='number of processes for multi-process training')


def add_predict_opts(parser):
    """Predicting options"""
    group = parser.add_argument_group("retrocomposer_predict")
    group.add('--beam_size', type=int, default=50, help='beam search size for rnn decoding')
