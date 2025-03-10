from typing import List, Any
import torch
import treetensor.torch as ttorch
from ding.utils.data import default_collate
from ding.torch_utils import to_tensor, to_ndarray, unsqueeze, squeeze


def default_preprocess_learn(
        data: List[Any],
        use_priority_IS_weight: bool = False,
        use_priority: bool = False,
        use_nstep: bool = False,
        ignore_done: bool = False,
) -> dict:
    # data preprocess
    data = default_collate(data)
    if ignore_done:
        data['done'] = torch.zeros_like(data['done']).float()
    else:
        data['done'] = data['done'].float()
    if use_priority_IS_weight:
        assert use_priority, "Use IS Weight correction, but Priority is not used."
    if use_priority and use_priority_IS_weight:
        if 'priority_IS' in data:
            data['weight'] = data['priority_IS']
        else:  # for compability
            data['weight'] = data['IS']
    else:
        data['weight'] = data.get('weight', None)
    if use_nstep:
        # Reward reshaping for n-step
        reward = data['reward']
        if len(reward.shape) == 1:
            reward = reward.unsqueeze(1)
        # reward: (batch_size, nstep) -> (nstep, batch_size)
        data['reward'] = reward.permute(1, 0).contiguous()

    return data


def single_env_forward_wrapper(forward_fn):

    def _forward(obs):
        obs = {0: unsqueeze(to_tensor(obs))}
        action = forward_fn(obs)[0]['action']
        action = to_ndarray(squeeze(action))
        return action

    return _forward


def single_env_forward_wrapper_ttorch(forward_fn, cuda=True):

    def _forward(obs):
        # unsqueeze means add batch dim, i.e. (O, ) -> (1, O)
        obs = ttorch.as_tensor(obs).unsqueeze(0)
        if cuda and torch.cuda.is_available():
            obs = obs.cuda()
        action = forward_fn(obs).action
        # squeeze means delete batch dim, i.e. (1, A) -> (A, )
        action = action.squeeze(0).cpu().numpy()
        return action

    return _forward
