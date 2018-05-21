import gym
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu

from rlkit.core import logger

from rlkit.envs.farmer import set_farm_port

import joblib
import argparse


def continue_experiment(args):
    logger.add_text_output('./d_text.txt')
    logger.add_tabular_output('./d_tabular.txt')
    logger.set_snapshot_dir('./snaps')

    extra = joblib.load(args.extra)
    
    algorithm = extra['algorithm']
    algorithm.farmlist_base = [('0.0.0.0', 15)]
    algorithm.refarm()

    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('extra', type=str,
                        help='path to the extra_data.pkl file')
    parser.add_argument('port', type=int,
                        help='farm port')
    args = parser.parse_args()
    print(args.extra)
    print(args.port)

    set_farm_port(args.port)

    setup_logger('name-of-experiment')
    continue_experiment(args)
