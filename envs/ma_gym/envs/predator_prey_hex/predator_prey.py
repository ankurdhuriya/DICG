from select import select
import gym
from gym import spaces
from ..utils.action_space import MultiAgentActionSpace
from ..utils.draw_hex import draw_grid

import random
import copy
import numpy as np

class PredatorPrey(gym.Env):
    '''
    Predator-prey involves a grid world with hexagonal cells (in a odd-q vertical layout shoves odd columns down).
    Agents have a 5x5 view and select one of seven actions ∈ {Top, TopRight, BottomRight, Bottom, BottomLeft, TopLeft} at each time step.
    Prey move according to selecting a uniformly random action at each time step.

    We define the “catching” of a prey as when the prey is within the cardinal direction of at least one predator.
    Each agent’s observation includes its own coordinates, agent ID, and the coordinates of the prey relative
    to itself, if observed.

    We modify the general predator-prey, such that a positive reward is given only if multiple predators catch a prey
    simultaneously, requiring a higher degree of cooperation. The predators get a team reward of 1 if two or more
    catch a prey at the same time, but they are given negative reward −P.We experimented with three varying P vales,
    where P = 0.5, 1.0, 1.5.

    The terminating condition of this task is when all preys are caught by more than one predator.
    For every new episodes , preys are initialized into random locations. Also, preys never move by themself into
    predator's neighbourhood
    '''

    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, grid_shape=(6, 8), n_agents=4, n_preys=2, prey_move_probs=(0.175, 0.175, 0.175, 0.175, 0.3),
                 penalty=-1.25, step_cost=-0.01, prey_capture_reward=10, max_steps=100) -> None:
        super(PredatorPrey).__init__()

        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self.n_preys = n_preys
        self._max_steps = max_steps
        self._step_count = None
        self._penalty = penalty
        self._step_cost = step_cost
        self._prey_capture_reward = prey_capture_reward
        self._agent_view_mask = (5, 5)

        self.action_space = MultiAgentActionSpace([spaces.Discrete(7) for _ in range(self.n_agents)])
        self.agent_pos = {_: None for _ in range(self.n_agents)}
        self.prey_pos = {_: None for _ in range(self.n_preys)}
        self._prey_alive = None

        self._base_grid = self.__create_grid()  # with no agents
        self._full_obs = self.__create_grid()
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._prey_move_probs = prey_move_probs
        self.viewer = None

        self._total_episode_reward = None
    
    def reset(self):
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_pos = {}
        self.prey_pos = {}

        self.__init_full_obs()
        self._step_count = 0
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._prey_alive = [True for _ in range(self.n_preys)]
        
        return self.get_agent_obs()
    
    def step(self, agents_action):
        self._step_count += 1
        rewards = [self._step_cost for _ in range(self.n_agents)]

        for agent_i, action in enumerate(agents_action):
            if not (self._agent_dones[agent_i]):
                self.__update_agent_pos(agent_i, action)

        for prey_i in range(self.n_preys):
            if self._prey_alive[prey_i]:
                predator_neighbour_count = self._neighbour_agents(self.prey_pos[prey_i])

                if predator_neighbour_count >= 1:
                    _reward = self._penalty if predator_neighbour_count == 1 else self._prey_capture_reward
                    self._prey_alive[prey_i] = (predator_neighbour_count == 1)

                    for agent_i in range(self.n_agents):
                        rewards[agent_i] += _reward

                prey_move = None
                if self._prey_alive[prey_i]:
                    for _ in range(5):
                        _move = np.random.choice(len(self._prey_move_probs), 1, p=self._prey_move_probs)[0]
                        if self._neighbour_agents(self.__next_pos(self.prey_pos[prey_i], _move)) == 0:
                            prey_move = _move
                            break
                        prey_move = 6 if prey_move is None else prey_move

                    self.__update_prey_pos(prey_i, prey_move)

        if (self._step_count >= self._max_steps) or (True not in self._prey_alive):
            for i in range(self.n_agents):
                self._agent_dones[i] = True
        
        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]

        return self.get_agent_obs(), rewards, self._agent_dones, {'prey_alive': self._prey_alive}

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    def __next_pos(self, pos, move):
        assert 0<=move<=6, "invalid move"
        assert self.is_valid(pos), "invalid position"

        d = {0:self.__move_top, 
            1:self.__move_topright,
            2:self.__move_bottomright,
            3:self.__move_bottom,
            4:self.__move_bottomleft,
            5:self.__move_topleft}

        if move == 6:
            return pos
        
        return d[move](pos)

    def __update_agent_pos(self, agent_i, action):
        curr_pos = copy.copy(self.agent_pos[agent_i])
    
        next_pos = self.__next_pos(curr_pos, action)

        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.agent_pos[agent_i] = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
            self.__update_agent_view(agent_i)

    def __update_prey_pos(self, prey_i, prey_move):
        curr_pos = copy.copy(self.prey_pos[prey_i])
        if self._prey_alive[prey_i]:
            next_pos = self.__next_pos(curr_pos, prey_move)

            if next_pos is not None and self._is_cell_vacant(next_pos):
                self.prey_pos[prey_i] = next_pos
                self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
                self.__update_prey_view(prey_i)
            else:
                pass

    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'])

    def get_agent_obs(self):
        _obs = []
        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]
            _agent_i_obs = [pos[0] / self._grid_shape[0], pos[1] / (self._grid_shape[1] - 1)]  # coordinates
            # print(f'agent {agent_i + 1} pos {pos} agent_obs {_agent_i_obs}')

            _prey_pos = []

            for i in range(6):
                _prey_pos.append(self.find_coords_in_direction(pos, i, 4))
            
            _agent_i_obs += np.array(_prey_pos).flatten().tolist() # adding prey po
            _agent_i_obs += [self._step_count / self._max_steps]  # adding time
            _obs.append(_agent_i_obs)
        
        return _obs

    def __init_full_obs(self):
        self._full_obs = self.__create_grid()
    
        for agent_i in range(self.n_agents):
            while True:
                pos = self.__get_random_position()
                if self.__is_cell_vacant(pos):
                    self.agent_pos[agent_i] = pos
                    break
            self.__update_agent_view(agent_i)

        for prey_i in range(self.n_preys):
            while True:
                pos = [random.randint(0, self._grid_shape[0] - 1), random.randint(0, self._grid_shape[1] - 1)]
                if self.__is_cell_vacant(pos) and (self._neighbour_agents(pos) == 0):
                    self.prey_pos[prey_i] = pos
                    break
            self.__update_prey_view(prey_i)

        self.__draw_base_img()
        self._base_img.show()

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape, self._full_obs, fill='white')

    def find_coords_in_direction(self, pos, direction, distance):
        coords = [0]*distance

        for i in range(distance):
            next_pos = self.__next_pos(pos, direction)
            if next_pos:
                pos = next_pos
                if PRE_IDS['prey'] in self._full_obs[pos[0]][pos[1]]:
                    coords[i] = 1
            else:
                break
        
        return coords


    def __move_top(self, pos):
        row, col = pos[0], pos[1]
        next_pos = [row, col-1]
        return next_pos if self.is_valid(next_pos) else None
    
    def __move_bottom(self, pos):
        row, col = pos[0], pos[1]
        next_pos = [row, col+1]
        return next_pos if self.is_valid(next_pos) else None
    
    def __move_bottomleft(self, pos):
        row, col = pos[0], pos[1]
        if row % 2 != 0 :
            col += 1
        next_pos = [row - 1, col]
        return next_pos if self.is_valid(next_pos) else None
    
    def __move_topleft(self, pos):
        row, col = pos[0], pos[1]
        if row % 2 != 0 :
            col += 1
        next_pos = [row - 1, col - 1]
        return next_pos if self.is_valid(next_pos) else None
    
    def __move_bottomright(self, pos):
        row, col = pos[0], pos[1]
        if row % 2 != 0 :
            col += 1
        next_pos = [row + 1, col]
        return next_pos if self.is_valid(next_pos) else None
    
    def __move_topright(self, pos):
        row, col = pos[0], pos[1]
        if row % 2 != 0 :
            col += 1
        next_pos = [row + 1, col - 1]
        return next_pos if self.is_valid(next_pos) else None

    def _neighbours_coords(self, pos):
        row, col = pos[0], pos[1]
        top = (row, col - 1)
        bottom = (row, col + 1)

        if row % 2 != 0 :
            col += 1
        bottomleft = (row - 1, col)
        topleft = (row - 1, col - 1)
        bottomright = (row + 1, col)
        topright = (row + 1, col - 1)

        coords = [top, topright, bottomright, bottom, bottomleft, topleft]
        return [coord if self.is_valid(coord) else None for coord in coords]

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _neighbour_agents(self, pos):
        neighbour_coords = self._neighbours_coords(pos)
        agent_count = 0
        for coord in neighbour_coords:
            if coord and PRE_IDS['agent'] in self._full_obs[coord[0]][coord[1]]:
                agent_count += 1      
        return agent_count

    def __update_agent_view(self, agent_i):
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['agent'] + str(agent_i + 1)

    def __update_prey_view(self, prey_i):
        self._full_obs[self.prey_pos[prey_i][0]][self.prey_pos[prey_i][1]] = PRE_IDS['prey'] + str(prey_i + 1)

    def __is_cell_vacant(self, pos):
        return self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty']

    def __get_random_position(self):
        return [random.randint(0, self._grid_shape[0] - 1), random.randint(0, self._grid_shape[1] - 1)]

    def __create_grid(self):
        _grid = [[PRE_IDS['empty'] for _ in range(self._grid_shape[0])] for row in range(self._grid_shape[1])]
        return _grid

    pass

PRE_IDS = {
        'empty': '0',
        'agent': 'A',
        'prey': 'P'
    }