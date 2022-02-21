from gym_connect4.envs.connect4_env import Connect4Env
import matplotlib.pyplot as plt

a = Connect4Env({"width": 7, "height": 6, "connect": 4})


def game(env, game, start, n):
    """perform an alternate game"""
    for k in range(start - 1, n + start):
        player, action = 1 + k % 2, 1 + k % 2
        obs, reward, done, info = a.step({player: action})
        mes = "" if not done["__all__"] else "#done"
        print(f"G{game}, Step {k} player {player} plays {action}, reward: {reward} {mes}")


game(a, 1, 1, 10)
a.print_board()
a.reset()
game(a, 2, 2, 10)
a.print_board()
