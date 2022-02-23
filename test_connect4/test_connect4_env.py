from gym_connect4.envs.connect4_env import Connect4Env

a = Connect4Env({"width": 7, "height": 6, "connect": 4,
                 "verbose": True, "visualization": True})


def game(env, start, n):
    """perform an alternate game"""
    for k in range(n):
        player, action = 1 + (k + start + 1) % 2, 1 + k % 2
        env.step({player: action})


game(a, 1, 10)
a.reset()
game(a, 2, 10)
a.reset()
a.step({1: 1})
a.step({2: 1})
a.step({1: 2})
a.step({2: 2})
a.step({1: 3})
a.step({2: 3})
a.step({1: 5})
a.step({2: 5})
a.step({1: 4})
a.reset()



