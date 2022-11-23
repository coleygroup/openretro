def add_model_opts(parser):
    """Model options"""
    group = parser.add_argument_group("localretro_model")
    group.add("--attention_heads", type=int, default=8)
    group.add("--attention_layers", type=int, default=1)
    group.add("--batch_size", type=int, default=16)
    group.add("--edge_hidden_feats", type=int, default=64)
    group.add("--node_out_feats", type=int, default=320)
    group.add("--num_step_message_passing", type=int, default=6)


def add_train_opts(parser):
    """Training options"""
    group = parser.add_argument_group("localretro_train")
    group.add('--batch_size', type=int, default=16, help='Batch size of dataloader')
    group.add('--num_epochs', type=int, default=50, help='Maximum number of epochs for training')
    group.add('--patience', type=int, default=5, help='Patience for early stopping')
    group.add('--max_clip', type=int, default=20, help='Maximum number of gradient clip')
    group.add('--learning_rate', type=float, default=1e-4, help='Learning rate of optimizer')
    group.add('--weight_decay', type=float, default=1e-6, help='Weight decay of optimizer')
    group.add('--schedule_step', type=int, default=10, help='Step size of learning scheduler')
    group.add('--num_workers', type=int, default=0, help='Number of processes for data loading')
    group.add('--print_every', type=int, default=20, help='Print the training progress every X mini-batches')
