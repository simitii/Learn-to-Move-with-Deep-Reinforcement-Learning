"""
Example of running PyTorch implementation of DDPG on HalfCheetah.
"""
import gym

from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
from rlkit.torch.ddpg.ddpg import DDPG
import rlkit.torch.pytorch_util as ptu

from rlkit.envs.farmer import farmer as Farmer

from rlkit.core import logger

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
    environment = acq_remote_env(farmer)
    env = NormalizedBoxEnv(environment)

    es = OUStrategy(action_space=env.action_space)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    net_size = variant['net_size']

    qf = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[net_size, net_size],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[net_size, net_size],
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = DDPG(
        env,
        qf=qf,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=1000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            use_soft_update=True,
            tau=1e-2,
            batch_size=128,
            max_path_length=1000,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
        ),
        net_size=64,
    )
    setup_logger('name-of-experiment', variant=variant)
    experiment(variant)
