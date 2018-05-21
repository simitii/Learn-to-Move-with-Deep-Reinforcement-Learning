from rlkit.samplers.util import rollout
from rlkit.torch.core import PyTorchModule
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import joblib
from rlkit.core import logger

from rlkit.envs.farmer import farmer as Farmer



def simulate_policy(args):
    data = joblib.load(args.file)
    policy = data['policy']
    env = data['env']
    print("Policy loaded")

    farmer = Farmer([('0.0.0.0', 1)])
    env_to_sim = farmer.force_acq_env()

    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    if isinstance(policy, PyTorchModule):
        policy.train(False)
    while True:
        path = rollout(
            env_to_sim,
            policy,
            max_path_length=args.H,
            animated=False,
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
