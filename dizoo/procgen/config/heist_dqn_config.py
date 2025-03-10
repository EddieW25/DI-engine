import torch.nn
from easydict import EasyDict

heist_dqn_config = dict(
    env=dict(
        env_id='heist',
        collector_env_num=4,
        evaluator_env_num=4,
        n_evaluator_episode=4,
        stop_value=10,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=[3, 64, 64],
            action_shape=15,
            encoder_hidden_size_list=[128, 128, 512],
            dueling=False,
        ),
        discount_factor=0.99,
        learn=dict(
            update_per_collect=20,
            batch_size=32,
            learning_rate=0.0005,
            target_update_freq=500,
        ),
        collect=dict(n_sample=100, ),
        eval=dict(evaluator=dict(eval_freq=5000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=250000,
            ),
            replay_buffer=dict(replay_buffer_size=100000, ),
        ),
    ),
)
heist_dqn_config = EasyDict(heist_dqn_config)
main_config = heist_dqn_config

heist_dqn_create_config = dict(
    env=dict(
        type='procgen',
        import_names=['dizoo.procgen.envs.procgen_env'],
    ),
    env_manager=dict(type='subprocess', ),
    policy=dict(type='dqn'),
)
heist_dqn_create_config = EasyDict(heist_dqn_create_config)
create_config = heist_dqn_create_config

if __name__ == '__main__':
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)