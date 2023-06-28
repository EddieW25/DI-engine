from easydict import EasyDict

heist_ppo_config = dict(
    env=dict(
        # frame_stack=4,
        is_train=True,
        env_id='heist',
        collector_env_num=4,
        evaluator_env_num=4,
        n_evaluator_episode=4,
        stop_value=10,
    ),
    policy=dict(
        cuda=False,
        action_space='discrete',
        model=dict(
            obs_shape=[3, 64, 64],
            action_shape=15,
            action_space='discrete',
            encoder_hidden_size_list=[32, 32, 64],
        ),
        learn=dict(
            update_per_collect=5,
            batch_size=64,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            learning_rate=0.0001,
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
heist_ppo_config = EasyDict(heist_ppo_config)
main_config = heist_ppo_config

heist_ppo_create_config = dict(
    env=dict(
        type='procgen',
        import_names=['dizoo.procgen.envs.procgen_env'],
    ),
    env_manager=dict(type='subprocess', ),
    policy=dict(type='ppo'),
)
heist_ppo_create_config = EasyDict(heist_ppo_create_config)
create_config = heist_ppo_create_config

if __name__ == '__main__':
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)