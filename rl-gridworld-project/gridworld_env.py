# gridworld_env.py
import numpy as np


class GridWorld:
    """
    A small GridWorld environment.
    Nothing fancy — just a 2D grid with start, goal, obstacles.
    """

    def __init__(
        self,
        width = 5,
        height = 5,
        start = (0,0),
        goal = (4,4),
        obstacles=None,
        max_steps = 50
    ):
        self.width   = width
        self.height  = height
        self.start   = start
        self.goal    = goal
        self.max_steps = max_steps

        if obstacles is None:
            # simple vertical wall to make the maze interesting
            obstacles = [(1,1),(2,1),(3,1)]

        self.obstacles = set(obstacles)
        self.n_states  = self.width * self.height
        self.n_actions = 4

        self.reset()


    def reset(self):
        self.agent_pos  = tuple(self.start)
        self.steps_taken = 0
        return self._state_to_idx(self.agent_pos)


    def _state_to_idx(self, pos):
        r, c = pos
        return r * self.width + c


    def _idx_to_state(self, idx):
        r = idx // self.width
        c = idx %  self.width
        return (r,c)


    def step(self, action):

        r, c = self.agent_pos

        if action == 0:      # UP
            nr, nc = r-1, c
        elif action == 1:    # DOWN
            nr, nc = r+1, c
        elif action == 2:    # LEFT
            nr, nc = r, c-1
        elif action == 3:    # RIGHT
            nr, nc = r, c+1
        else:
            raise ValueError("invalid action??")

        # collision check
        if (
            nr < 0 or nr >= self.height
            or nc < 0 or nc >= self.width
            or (nr,nc) in self.obstacles
        ):
            nr, nc = r, c   # bounce back

        self.agent_pos = (nr,nc)
        self.steps_taken += 1

        # reward logic
        if self.agent_pos == self.goal:
            reward = 10.0
            done = True
        else:
            reward = -1.0
            done = False

        if self.steps_taken >= self.max_steps:
            done = True

        return self._state_to_idx(self.agent_pos), reward, done, {}


    def render(self):
        """just print the grid as text"""
        g = [["." for _ in range(self.width)] for _ in range(self.height)]

        for (r,c) in self.obstacles:
            g[r][c] = "#"

        sr,sc = self.start
        gr,gc = self.goal
        g[sr][sc]="S"
        g[gr][gc]="G"

        ar,ac = self.agent_pos
        g[ar][ac] = "A"

        for row in g:
            print(" ".join(row))
        print()

