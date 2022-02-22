from gym_connect4.envs.connect4_env import Connect4Env
import matplotlib.pyplot as plt

a = Connect4Env({"width": 7, "height": 6, "connect": 4, "verbose": True})


def game(env, game, start, n):
    """perform an alternate game"""
    for k in range(n):
        player, action = 1 + (k + start + 1) % 2, 1 + k % 2
        obs, reward, done, info = a.step({player: action})


game(a, 1, 1, 10)
a.to_string()
a.reset()
game(a, 2, 2, 10)
a.to_string()
a.reset()
a.step({1: 1})
a.step({1: 1})
a.step({2: 1})
a.step({2: 1})
a.step({1: 1})
a.step({1: 1})
a.to_string()
