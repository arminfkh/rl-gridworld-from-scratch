class GridWorld:

    def __init__(self):
        self.rows = 5
        self.cols = 5
        self.start_position = (0, 0)
        self.goal_position = (4, 4)
        self.walls = {(1, 1), (1, 2), (3, 1), (3, 3)}
        self.agent_position = self.start_position
        self.actions = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

    def reset(self):
        self.agent_position = self.start_position
        return self.agent_position

    def step(self, action):
        agent_row, agent_col = self.agent_position
        action_row, action_col = self.actions[action]
        candidate_position = (agent_row + action_row, agent_col + action_col)

        if (
            0 <= candidate_position[0] < self.rows
            and 0 <= candidate_position[1] < self.cols
            and candidate_position not in self.walls
        ):
            self.agent_position = candidate_position
            if self.agent_position == self.goal_position:
                reward = 20
                terminated = True
            else:
                reward = -1
                terminated = False

        else:
            reward = -2
            terminated = False

        return self.agent_position, reward, terminated

    def render(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) == self.agent_position:
                    print("A", end=" ")
                elif (i, j) in self.walls:
                    print("#", end=" ")
                elif (i, j) == self.goal_position:
                    print("G", end=" ")
                else:
                    print(".", end=" ")
            print()
        print()
