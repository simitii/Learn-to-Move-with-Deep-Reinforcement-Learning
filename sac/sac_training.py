"""
Run PyTorch Soft Actor Critic on HalfCheetahEnv.

NOTE: You need PyTorch 0.3 or more (to have torch.distributions)
"""
import gym
import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.networks import FlattenMlp

from rlkit.core import logger

from rlkit.envs.farmer import farmer as Farmer
from rlkit.envs.farmer import set_farm_port

import traceback

farmlist_base = [('0.0.0.0', 34)]

def acq_remote_env(farmer):
    # acquire a remote environment
    while True:
        remote_env = farmer.acq_env()
        if remote_env == False:  # no free environment
            pass
        else:
            break
    remote_env.set_spaces()
    print('action space', remote_env.action_space)
    print('observation space', remote_env.observation_space)
    return remote_env



def experiment(variant):
    logger.add_text_output('./d_text.txt')
    logger.add_tabular_output('./d_tabular.txt')
    logger.set_snapshot_dir('./snaps')
    farmer = Farmer([('0.0.0.0', 1)])
    remote_env = farmer.force_acq_env()
    remote_env.set_spaces()
    env = NormalizedBoxEnv(remote_env)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    net_size = variant['net_size']
    qf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    algorithm = SoftActorCritic(
        env=env,
        training_env=env,
        policy=policy,
        qf=qf,
        vf=vf,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    import sys
    try:
        farm_port = int(sys.argv[1])
    except:
        farm_port = 20099
    set_farm_port(farm_port)

    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=1000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=999,
            discount=0.99,

            soft_target_tau=0.001,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,

            environment_farming=True,
            farmlist_base=farmlist_base,

            save_replay_buffer=True,
            save_algorithm=True,
            save_environment=True,
        ),
        net_size=64,
    )
    setup_logger('name-of-experiment', variant=variant)
    experiment(variant)
