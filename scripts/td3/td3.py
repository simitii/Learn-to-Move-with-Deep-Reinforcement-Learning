"""
This should results in an average return of ~3000 by the end of training.

Usually hits 3000 around epoch 80-100. Within a see, the performance will be
a bit noisy from one epoch to the next (occasionally dips dow to ~2000).

Note that one epoch = 5k steps, so 200 epochs = 1 million steps.
"""

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
from rlkit.torch.td3.td3 import TD3

from rlkit.envs.farmer import farmer as Farmer
from rlkit.envs.farmer import set_farm_port

from rlkit.core import logger

farmlist_base = [('0.0.0.0', 15)]


def experiment(variant):
    logger.add_text_output('./d_text.txt')
    logger.add_tabular_output('./d_tabular.txt')
    logger.set_snapshot_dir('./snaps')

    farmer = Farmer([('0.0.0.0', 1)])
    remote_env = farmer.force_acq_env()
    remote_env.set_spaces()
    env = NormalizedBoxEnv(remote_env)

    es = GaussianStrategy(
        action_space=env.action_space,
        max_sigma=0.1,
        min_sigma=0.1,  # Constant sigma
    )
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[256, 256],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[256, 256],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[256, 256],
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = TD3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_kwargs=dict(
            num_epochs=500,
            num_steps_per_epoch=2000,
            num_steps_per_eval=10000,
            max_path_length=1000,
            batch_size=128,
            discount=0.995,

            environment_farming=True,
            farmlist_base=farmlist_base,

            save_replay_buffer=True,
            save_algorithm=True,
            save_environment=True,
        ),
    )
    setup_logger('name-of-td3-experiment', variant=variant)
    experiment(variant)
