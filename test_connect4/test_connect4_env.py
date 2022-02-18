from gym_connect4.envs.connect4_env import Connect4Env


k = 2
n = 10
a = Connect4Env({"width": 7, "height": 6, "connect": 4})
for k in range(n):
    print("Step", k)
    player, action = 1 + k % 2, 1 + k % 2
    obs, reward, done, info = a.step({player: action})
    print(f"player: {player} , reward: {reward}")
    if done["__all__"]:
        print(f"# the game is done")
a.render(mode="human")
