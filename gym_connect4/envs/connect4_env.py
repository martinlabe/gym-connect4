import gym
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class Connect4Env(MultiAgentEnv):

    ## RESOURCES ##############################################################
    # MultiAgentEnv examples from rllib
    # https://github.com/ray-project/ray/blob/master/rllib/examples/env/multi_agent.py
    # Towards Data Science article
    # https://towardsdatascience.com/beginners-guide-to-custom-environments-in-openai-s-gym-989371673952

    ## GYM ####################################################################
    def __init__(self, conf):
        """
        define the action and observation spaces

        :conf:   dict with the configuration parameters
        """
        self.type = np.bool
        self.size = (conf["height"], conf["width"])
        self.connect = conf["connect"]
        self.done_p = np.zeros(2, dtype=bool)
        self.grid_p1 = np.zeros(self.size, dtype=self.type)
        self.grid_p2 = np.zeros(self.size, dtype=self.type)
        self.action_space = gym.spaces.Discrete(self.size[1])
        self.observation_space = gym.spaces.Box(low=False, high=True,
                                                shape=(self.size[0],
                                                       self.size[1],
                                                       2),
                                                dtype=self.type)
        self.test_mode = 1 if ("test_mode", 1) in conf.items() else 0
        self.done = {"__all__": False}

    def step(self, actions):
        """
        compute the consequences of a step

        :action: the action taken by the agent
        :return:
            :state:  observation_space after the action
            :reward: reward given for the action
            :done:   a boolean saying if it reached an endpoint
            :info:   a dictionary that can be used in bug fixing
        """
        player, action = list(actions.items())[0]
        reward = 0
        # if the game is over
        if self.done_p.all():
            self.done["__all__"] = True
        # if the opponent won
        elif self.done_p.any():
            reward = - 100
            self.done_p[:] = True
            self.done["__all__"] = True
        else:
            position = self.play(player, action)
            # if the action is illegal
            if position is None:
                # if it is because the grid is full
                if self.get_grid().all():
                    self.done["__all__"] = True
                # if it is because of the rules
                else:
                    reward = -1
            # if the action is legal
            else:
                # if it leads to wining
                if self.win(player, position):
                    self.done_p[player - 1] = True
                    reward = 100
                # if it's not
                else:
                    reward = -1
        return {player: self.render()}, {player: reward}, self.done, {}

    def reset(self):
        """restart the environment"""
        self.done_p1 = False
        self.done_p2 = False
        self.grid_p1[:] = 0
        self.grid_p2[:] = 0
        self.done["__all__"] = False
        return {1: self.render()}

    ## GETTERS ################################################################
    def get_grid(self):
        """return the game grid"""
        return self.grid_p1 | self.grid_p2

    def get_grid_p(self, player):
        """return the grid of the player"""
        return self.grid_p1 if player == 1 else self.grid_p2

    ## MY GAME ################################################################
    def render(self, mode='machine'):
        """
        generate the input for the network

        :return: a (6, 7, 2) numpy array of self.type
        """
        if mode == 'human':
            self.print_board()
            return

        return np.dstack((self.grid_p1, self.grid_p2))

    def play(self, player, column):
        """do the player move"""
        # if it's an invalid move (out of the grid)
        if column >= self.size[1] or column < 0:
            return None
        # if it is an invalid move (invalid rule)
        if self.grid_p1[0, column] or self.grid_p2[0, column]:
            return None
        # else
        grid = self.get_grid()
        line = self.size[0] - np.argmax(np.flip(
            np.logical_not(grid[:, column]))) - 1
        if player == 1:
            self.grid_p1[line][column] = True
        else:
            self.grid_p2[line][column] = True
        self.last_move = (line, column)
        return line, column

    def win(self, player, position):
        """returns if the player won"""
        assert player == 1 or player == 2, "player is incorrect"
        assert type(position) is tuple and len(position) == 2, "position incorrect"
        grid_p = self.get_grid_p(player)

        # check vertically
        if self.size[0] - position[0] >= self.connect:
            if grid_p[position[0]: position[0] + self.connect,
               position[1]].all():
                return True
        # check horizontally
        start = max(position[1] - (self.connect - 1), 0)
        end = min(position[1], self.size[1] - self.connect)
        for j in range(start, end + 1):
            if grid_p[position[0]][j: j + self.connect].all():
                return True

        def check_diagonally(grid_pp, pos):
            """check diagonally from up left to bottom right"""
            count_up = min(pos[0], self.connect - 1)
            count_left = min(pos[1], self.connect - 1)
            count_lu = min(count_left, count_up)
            count_right = min(self.size[1] - pos[1] - 1, self.connect - 1)
            count_bottom = min(self.size[0] - pos[0] - 1, self.connect - 1)
            count_br = min(count_right, count_bottom)
            align = count_lu + count_br + 1
            count = 0
            for k in range(0, align):
                if align < self.connect:
                    break
                if grid_pp[pos[0] - count_lu + k, pos[1] - count_lu + k]:
                    count += 1
                else:
                    count = 0
                if count >= self.connect:
                    return True

        # check diagonally 3pi/4
        if check_diagonally(grid_p, position):
            return True
        # check diagonally pi/4 by using the (0x) symmetry
        if check_diagonally(np.fliplr(grid_p),
                            (position[0], self.size[1] - position[1] - 1)):
            return True

        return False

    ## EXTRA ##################################################################
    def print_board(self):
        """pretty print of the board"""
        pieces = ['🟦', '🟡', '🔴']
        buffer = ""
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if self.grid_p1[i][j]:
                    buffer += pieces[1]
                elif self.grid_p2[i][j]:
                    buffer += pieces[2]
                else:
                    buffer += pieces[0]
            buffer += '\n'
        print(buffer)
