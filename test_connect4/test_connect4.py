from gym_connect4.envs.connect4_env import Connect4Env


def test(actions, verbose=True):
    """test a list of instructions"""
    a = Connect4Env({"width": 7, "height": 6, "connect": 4})
    n = len(actions)
    res = True
    for i, (player, action) in enumerate(actions):
        pos = a.play(player, action)
        out = a.win(player, pos)
        # intermediary step
        if i != n - 1:
            if out:
                res = False
        # final step
        else:
            if not out:
                res = False
    if verbose:
        print(res)
        a.print_board()
    elif not res:
        print(res)


test([(1, 0), (1, 0), (1, 0), (1, 0)])
test([(2, 0), (2, 0), (1, 0), (1, 0), (1, 0), (1, 0)])
test([(2, 2), (2, 4), (2, 5), (2, 3)])
test([(1, 0), (1, 0), (1, 0), (2, 0), (1, 1), (1, 1), (2, 1), (1, 2), (2, 2),
      (2, 3)])
test([(2, 3), (1, 4), (2, 4), (1, 5), (1, 5), (2, 5), (1, 6), (1, 6), (1, 6),
      (2, 6)])
test([(1, 1), (2, 1), (1, 2), (1, 2), (2, 2), (2, 3), (1, 3), (2, 3), (2, 3),
      (1, 4), (2, 4), (1, 4), (1, 4), (2, 4)])
