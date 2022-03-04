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
        def __get_attribute(name):
            """return True if the attribute is set to True"""
            return name in list(conf.keys()) and conf[name]

        def __get_observation_space(compact=True):
            """choose to work either with (6, 7, 2) boolean or (6, 7) bytes"""
            if compact:
                return gym.spaces.Box(low=-1, high=1,
                                      shape=(self.size[0],
                                             self.size[1]),
                                      dtype=np.byte)
            else:
                return gym.spaces.Box(low=False, high=True,
                                      shape=(self.size[0],
                                             self.size[1],
                                             2),
                                      dtype=bool)

        # public
        ## for the game
        self.player = 0
        self.done = False
        self.state = ""
        ## for the config
        self.size = (conf["height"], conf["width"])
        self.connect = conf["connect"]
        ## for modes
        self.verbose = __get_attribute("verbose")
        self.visualization = __get_attribute("visualization")
        # for the class
        self.__count = 0
        # choose (6, 7, 2) or (6, 7), also change low and high is obsspace
        self.__grid_p1 = np.zeros(self.size, dtype=bool)
        self.__grid_p2 = np.zeros(self.size, dtype=bool)
        ## for gym
        self.action_space = gym.spaces.Discrete(self.size[1])
        self.observation_space = __get_observation_space()


    def step(self, action_dict):
        """
        compute the consequences of a step

        :action: the action taken by the agent
        :return:
            :state:  observation_space after the action
            :reward: reward given for the action
            :done:   a boolean saying if it reached an endpoint
            :info:   a dictionary that can be used in bug fixing
        """
        assert len(action_dict.items()) == 1, "Alternated game: wrong input dict"

        player, action = list(action_dict.items())[0]
        self.__count += 1

        WIN, DRAW, LOSS = 100, 0, -100,
        PLAY, WRONG, OVER = 0, -10, 0
        obs, rew, done, info = {}, \
                               {}, \
                               {"__all__": False}, \
                               {}

        def verbose(message):
            """print the step message to debug"""
            if self.verbose:
                v_player, v_action = list(action_dict.items())[0]
                print(f"Step {self.__count} actions: [{v_player}, {v_action + 1}], reward: {rew} #{message}")

        # if the game is over
        if self.done:
            obs = {}
            rew[player] = OVER
            done["__all__"] = True
            verbose(f"OVER: the game is over.")
            return obs, rew, done, info
        # if it is not the turn
        elif self.player == player:
            obs[self.get_other_player(player)] = self.board(self.get_other_player(player))
            rew[player] = WRONG
            done["__all__"] = self.done
            info[self.get_other_player(player)] = "WRONG"
            verbose(f"{info[player]}: not the turn of player {player}")
            return obs, rew, done, info
        # else let's play
        position = self.play(player, action)
        # update the last player
        self.player = player
        obs[self.get_other_player(player)] = self.board(self.get_other_player(player))
        # if the action is illegal (the move)
        if position is None:
            # if it is because the grid is full
            if self.get_grid().all():
                rew[player] = DRAW
                rew[self.get_other_player(player)] = DRAW
                done["__all__"] = True
                info[self.get_other_player(player)] = "DRAW"
                verbose(f"{info[self.get_other_player(player)]}: player {player} is facing a full grid.")
            # if it is because of the rules
            else:
                rew[player] = WRONG
                #done["__all__"] = True
                info[self.get_other_player(player)] = "WRONG"
                verbose(f"{info[self.get_other_player(player)]}: player {player} tried to play {action + 1}.")
        # if the action is legal
        else:
            # if it leads to wining
            if self.win(player, position):
                rew[player] = WIN
                rew[self.get_other_player(player)] = LOSS
                done["__all__"] = True
                info[self.get_other_player(player)] = "WIN"
                verbose(f"{info[self.get_other_player(player)]}: player {player} won")
            # if it's not
            else:
                rew[player] = PLAY
                done["__all__"] = False
                info[self.get_other_player(player)] = "PLAY"
                verbose("")

        # inform the environment if the game is done
        self.done = done["__all__"]
        self.state = info[self.get_other_player(player)]
        return obs, rew, done, info

    def reset(self):
        """restart the environment"""
        if self.visualization:
            self.to_string()
        self.done = False
        self.player = 0
        self.__count = 0
        self.__grid_p1[:] = 0
        self.__grid_p2[:] = 0
        return {1: self.board(1)}

    ## GETTERS ################################################################
    def get_grid(self):
        """return the game grid"""
        return self.__grid_p1 | self.__grid_p2

    def get_grid_p(self, player):
        """return the grid of the player"""
        return self.__grid_p1 if player == 1 else self.__grid_p2

    @staticmethod
    def get_other_player(player):
        return 1 if player == 2 else 2

    ## MY GAME ################################################################
    def board(self, player):
        """
        generate the input for the network

        :return: a (6, 7, 2) numpy array of self.type
        """
        res = None
        grid_pa = self.__grid_p1 if player == 1 else self.__grid_p2
        grid_pb = self.__grid_p2 if player == 1 else self.__grid_p1
        # return a (6, 7) byte array with agent 1 and other -1
        if len(self.observation_space.shape) == 2:
            res = np.zeros(self.observation_space.shape,
                           dtype=self.observation_space.dtype)
            res[grid_pa == 1] = 1
            res[grid_pb == 1] = -1
        # return a (6, 7, 2) bool array with agent True and other False
        else:
            res = np.dstack((grid_pa, grid_pb))
        return res

    def render(self, mode="human"):
        """generate the render of the game"""
        if mode == 'human':
            self.to_string()
        elif mode == 'machine':
            return self.to_image()

    def play(self, player, column):
        """do the player move"""
        # if it's an invalid move (out of the grid)
        if column >= self.size[1] or column < 0:
            return None
        # if it is an invalid move (invalid rule)
        if self.__grid_p1[0, column] or self.__grid_p2[0, column]:
            return None
        # else we play the move
        grid = self.get_grid()
        line = self.size[0] - np.argmax(np.flip(
            np.logical_not(grid[:, column]))) - 1
        if player == 1:
            self.__grid_p1[line][column] = True
        else:
            self.__grid_p2[line][column] = True
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
    def to_string(self):
        """pretty print of the board"""
        pieces = ['🟦', '🟡', '🔴']
        buffer = ""
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if self.__grid_p1[i][j]:
                    buffer += pieces[1]
                elif self.__grid_p2[i][j]:
                    buffer += pieces[2]
                else:
                    buffer += pieces[0]
            buffer += '\n'
        buffer += "1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣"
        print(buffer)

    def to_image(self):
        """
        generate an image of the game
        """
        res = np.zeros(shape=(self.size[0], self.size[1], 3), dtype=np.uint8)

        # color used
        YELLOW = (255, 255, 0)
        RED = (255, 0, 0)
        BLUE = (0, 0, 255)

        # filling the image
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if self.__grid_p1[i, j]:
                    res[i, j] = YELLOW
                elif self.__grid_p2[i, j]:
                    res[i, j] = RED
                else:
                    res[i, j] = BLUE
        return res
