def add_model_opts(parser):
    """Model options"""
    group = parser.add_argument_group("retroxpert_model")
    group.add("--typed", help="whether reaction class is given", action="store_true")


def add_preprocess_opts(parser):
    """Preprocessing options"""
    group = parser.add_argument_group("retroxpert_preprocess")
    group.add("--min_freq", help="minimum frequency for patterns to be kept", type=int, default=2)
