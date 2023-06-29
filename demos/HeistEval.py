import os
import gym
import torch
from tensorboardX import SummaryWriter
from easydict import EasyDict

from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
from ding.envs import BaseEnvManager, DingEnvWrapper
from ding.policy import DQNPolicy
from ding.model import DQN
from ding.utils import set_pkg_seed
from ding.rl_utils import get_epsilon_greedy_fn
from dizoo.procgen.config.heist_dqn_config import main_config, create_config

def wrapped_coinrun_env():
    env = gym.make("procgen:procgen-heist-v0")
    env.seed(0)  # Set the seed directly on the environment
    return DingEnvWrapper(env, EasyDict(env_wrapper='default'))


def main(cfg, seed=0):
    cfg['exp_name'] = 'heiset_dqn_eval'
    cfg = compile_config(
        cfg,
        BaseEnvManager,
        DQNPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        AdvancedReplayBuffer,
        save_cfg=True
    )

    cfg.policy.load_path = '/home/liwen/PycharmProjects/DI-engine/dizoo/procgen/config/default_experiment_230626_010946/ckpt/iteration_330000.pth.tar' #Heist DQN experiment last iteration

    # build multiple environments and use env_manager to manage them
    evaluator_env_num = cfg.env.evaluator_env_num
    evaluator_env = BaseEnvManager(env_fn=[wrapped_coinrun_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager)

    # switch save replay interface
    # evaluator_env.enable_save_replay(cfg.env.replay_path)
    evaluator_env.enable_save_replay(replay_path='./Heist_dqn_eval/video')

    # Set random seed for all package and instance
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    # Set up RL Policy
    model = DQN(**cfg.policy.model)
    policy = DQNPolicy(cfg.policy, model=model)
    policy.eval_mode.load_state_dict(torch.load(cfg.policy.load_path, map_location='cpu'))

    # Evaluate
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator.eval()

if __name__ == "__main__":
    main(main_config)