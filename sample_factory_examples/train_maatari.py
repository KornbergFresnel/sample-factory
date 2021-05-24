from os import stat_result
import sys
from torch import nn
from sample_factory.algorithms.utils.arguments import arg_parser, parse_args
from sample_factory.run_algorithm import run_algorithm
from sample_factory.algorithms.appo.model_utils import register_custom_encoder, EncoderBase, get_obs_shape, nonlinearity
from sample_factory.algorithms.utils.pytorch_utils import calc_num_elements


class CustomEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)

        obs_shape = get_obs_shape(obs_space)
        # print("----------- obs shape:", obs_shape.obs)

        conv_layers = [
            nn.Conv2d(obs_shape.obs[-1], 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nonlinearity(cfg),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(4, 4, kernel_size=4, stride=1, padding=1),
            # nn.BatchNorm2d(4),
            # nonlinearity(cfg),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        self.conv_head = nn.Sequential(*conv_layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape.obs, format='NHWC')

        self.init_fc_blocks(self.conv_head_out_size)

    def forward(self, obs_dict):
        # we always work with dictionary observations. Primary observation is available with the key 'obs'
        main_obs = obs_dict['obs']

        main_obs = main_obs.permute(0, 3, 1, 2)
        # print("----------- shape of main_obs:", main_obs.shape)
        x = self.conv_head(main_obs)
        x = x.reshape((-1, self.conv_head_out_size))

        # forward pass through configurable fully connected blocks immediately after the encoder
        x = self.forward_fc_blocks(x)
        return x


def custom_parse_args(argv=None, evaluation=False):
    """
    Parse default SampleFactory arguments and add user-defined arguments on top.
    Allow to override argv for unit tests. Default value (None) means use sys.argv.
    Setting the evaluation flag to True adds additional CLI arguments for evaluating the policy (see the enjoy_ script).

    """
    parser = arg_parser(argv, evaluation=evaluation)
    cfg = parse_args(argv=argv, evaluation=evaluation, parser=parser)
    return cfg


def main():
    register_custom_encoder('custom_env_encoder', CustomEncoder)
    cfg = custom_parse_args()
    status = run_algorithm(cfg)
    return status


if __name__ == '__main__':
    # python -m sample_factory.examples.train_maatari --algo=APPO --env=maatari_pong --experiment=test_maatari_pong --help
    sys.exit(main())
