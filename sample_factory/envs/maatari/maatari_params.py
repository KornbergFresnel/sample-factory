def maatari_override_defaults(env, parser):
    """
    Override default argument values for this family of environments.
    All experiments for environments from my_custom_env_ family will have these parameters unless
    different values are passed from command line.

    """
    parser.set_defaults(
        encoder_custom='custom_env_encoder'
    )


def add_maatari_env_args(env, parser):
    # add custom args here
    parser.add_argument('--resolution', type=int, default=12, help="Resolution")