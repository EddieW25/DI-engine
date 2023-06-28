import gym # Load the gym library, which is used to standardize the reinforcement learning environment
import torch # Load the PyTorch library for loading the Tensor model and defining the computing network
from easydict import EasyDict # Load EasyDict for instantiating configuration files
from ding.config import compile_config # Load configuration related components in DI-engine config module
from ding.envs import DingEnvWrapper # Load environment related components in DI-engine env module
from ding.policy import DQNPolicy, single_env_forward_wrapper # Load policy-related components in DI-engine policy module
from ding.model import DQN # Load model related components in DI-engine model module
from dizoo.box2d.lunarlander.config.lunarlander_dqn_config import main_config, create_config # Load DI-zoo lunarlander environment and DQN algorithm related configurations


def main(main_config: EasyDict, create_config: EasyDict, ckpt_path: str):
    main_config.exp_name = 'lunarlander_dqn_deploy' # Set the name of the experiment to be run in this deployment, which is the name of the project folder to be created
    cfg = compile_config(main_config, create_cfg=create_config, auto=True) # Compile and generate all configurations
    env = DingEnvWrapper(gym.make(cfg.env.env_id), EasyDict(env_wrapper='default')) # Add the DI-engine environment decorator upon the gym's environment instance
    env.enable_save_replay(replay_path='../lunarlander_dqn_deploy/video') # Enable the video recording of the environment and set the video saving folder
    model = DQN(**cfg.policy.model) # Import model configuration, instantiate DQN model
    state_dict = torch.load(ckpt_path, map_location='cpu') # Load model parameters from file
    model.load_state_dict(state_dict['model']) # Load model parameters into the model
    policy = DQNPolicy(cfg.policy, model=model).eval_mode # Import policy configuration, import model, instantiate DQN policy, and turn to evaluation mode
    forward_fn = single_env_forward_wrapper(policy.forward) # Use the strategy decorator of the simple environment to decorate the decision method of the DQN strategy
    obs = env.reset() # Reset the initialization environment to get the initial observations
    returns = 0. # Initialize total reward
    while True: # Let the agent's strategy and environment interact cyclically until the end
        action = forward_fn(obs) # According to the observed state, make a decision and generate action
        obs, rew, done, info = env.step(action) # Execute actions, interact with the environment, get the next observation state, the reward of this interaction, the signal of whether to end, and other information
        returns += rew # Cumulative reward return
        if done:
            break
    print(f'Deploy is finished, final epsiode return is: {returns}')

if __name__ == "__main__":
    main(main_config=main_config, create_config=create_config, ckpt_path='/mnt/c/Users/eddie/Downloads/final.pth.tar')