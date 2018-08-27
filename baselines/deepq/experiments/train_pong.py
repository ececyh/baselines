import gym

from baselines import deepq
from baselines.common.atari_wrappers import wrap_deepmind, ScaledFloatFrame


def main():
    env = gym.make("PongNoFrameskip-v4")
    env = ScaledFloatFrame(wrap_deepmind(env))
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True #True
    )
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=10000000,
        buffer_size=500000,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        train_freq=4,
        print_freq=1,
        learning_starts=10000,
        target_network_update_freq=10000,
        gamma=0.99,
        prioritized_replay=True #True
    )
    act.save("pong_model.pkl")
    env.close()


if __name__ == '__main__':
    main()
