import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

ACTIONS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)} # Step up, down, left, right
GRID_SIZE = 10 # GRID_SIZE-by-GRID_SIZE square grid
START = (0, 0)
GOAL = (GRID_SIZE - 1, GRID_SIZE - 1)
TRAPS = {(3, 1), (4, 3), (2, 4), (0, 4), (6, 3), (6, 4), (7, 4), (3, 8), (9, 8), (8, 8), (7, 6)}
WALLS = {(0, 2), (2, 0), (2, 1), (4, 4), (3, 3), (6, 6), (6, 7), (6, 8), (6, 9), (7, 1), (8, 2), (9, 3)}


class TreasureHunt:
    def __init__(self):
        '''
        Numerical tile code:
            0 = empty tile
            1 = wall
            2 = agent
            3 = trap
            9 = treasure
        '''
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype = int)

        for pos in WALLS:
            self.grid[pos] = 1
        for trap in TRAPS:
            self.grid[trap] = 3
        self.grid[GOAL] = 9
        self.grid[START] = 2

        self.agent_pos = START
        self.steps = 0
        self.game_over = False
        self.generate_allowed_moves()

    def print_grid(self) -> None:
        print("Board at step 0 (\033[93mÅ\033[0m = agent, \033[91m╳\033[0m = trap, \033[92m◯\033[0m = treasure, ▉ = wall, (empty) = safe tile):")
        symbols = {0: " ", 1: "▉", 2: "\033[93mÅ\033[0m", 3: "\033[91m╳\033[0m", 9: "\033[92m◯\033[0m"}
        print("╭───┬" + "───┬" * (GRID_SIZE - 2) + "───╮")
        for row in self.grid:
            print("│ " + " │ ".join(symbols[c] for c in row) + " │")
            if not np.array_equal(row, self.grid[-1]):
                print("├───┼" + "───┼" * (GRID_SIZE - 2) + "───┤")
            else:
                print("╰───┴" + "───┴" * (GRID_SIZE - 2) + "───╯")

    def generate_allowed_moves(self) -> None:
        allowed = {}
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.grid[y, x] == 1: # Ignore wall tiles
                    continue
                moves = []
                for move, val in ACTIONS.items():
                    dy, dx = val
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < GRID_SIZE and 0 <= nx < GRID_SIZE: # Not allowed to step out of bounds
                        if self.grid[ny, nx] != 1: # Not allowed to move through walls
                            moves.append(move)
                allowed[(y, x)] = moves
        self.allowed_moves = allowed

    def step(self,
             action: str) -> tuple[int, int] and int:
        y, x = self.agent_pos
        self.grid[y, x] = 3 if (y, x) in TRAPS else 0 # Restore tile value upon agent leaving tile (otherwise tiles are overwritten with 2's)
        
        dy, dx = ACTIONS[action]
        ny, nx = y + dy, x + dx
        self.agent_pos = (ny, nx)
        self.steps += 1

        tile = self.grid[ny, nx]
        if tile == 9:
            reward = 10
            self.game_over = True
        elif tile == 3:
            reward = -10
            self.game_over = True
        else:
            reward = -1
        self.grid[ny, nx] = 2

        return self.agent_pos, reward
    
    @property
    def is_done(self) -> bool:
        return self.game_over or self.steps > 500
    
    def reset(self) -> None:
        for pos in WALLS:
            self.grid[pos] = 1
        for trap in TRAPS:
            self.grid[trap] = 3
        self.grid[GOAL] = 9
        self.grid[START] = 2

        self.agent_pos = START
        self.steps = 0
        self.game_over = False
        


class Agent:
    def __init__(self,
                 alpha: float = 0.12,
                 gamma: float = 0.95,
                 exploration_rate: float = 0.35,
                 exploration_reduction: float = 1e-4):
        self.alpha = alpha # Learning rate
        self.gamma = gamma # Discount factor, how much future rewards count
        self.exploration_rate = exploration_rate # Exploration rate
        self.exploration_reduction = exploration_reduction # Reduction of exploration rate per episode

        self.Q = {} # Q-table, maps (state, action) -> value, meaning the agent learns from both states and actions.
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                for action, (dy, dx) in ACTIONS.items(): # Initializing board with random moves
                    self.Q[((y, x), action)] = rng.random()

    def choose_action(self,
                      state: tuple[int, int],
                      allowed_moves: list[str]) -> str:
        if rng.random() < self.exploration_rate:
            return np.random.choice(allowed_moves)

        best_action = None
        best_q = -1e15
        for action in allowed_moves:
            q = self.Q.get((state, action), 0.0)
            if q >= best_q:
                best_q = q
                best_action = action
        return best_action

    def update(self,
               state: tuple[int, int],
               action: str,
               reward: int,
               next_state: tuple[int, int],
               next_allowed: list[str],
               done: bool) -> None:
        '''
        Bellman equation: Q(s,a) gets nudged toward the reward received plus the discounted value of
        the best action available next. On last step there's no future value, just the reward itself.
        '''
        if done:
            td_target = reward
        else:
            best_next_q = max(self.Q.get((next_state, a), 0.0) for a in next_allowed)
            td_target = reward + self.gamma * best_next_q

        key = (state, action)
        self.Q[key] = self.Q.get(key, 0.0) + self.alpha * (td_target - self.Q.get(key, 0.0))

    def reduce_exploration(self):
        self.exploration_rate = max(0.001, self.exploration_rate - self.exploration_reduction) # Reduces the exploration rate and puts a non-zero lower limit on it.


if __name__ == "__main__":
    NUM_EPISODES = 2000
    agent = Agent(0.12, 0.95, 0.35, 2e-4)
    step_history = []
    reward_history = []
    board = TreasureHunt()
    initial_Q = agent.Q.copy()
    board.print_grid()
    print()

    for episode in range(NUM_EPISODES):
        total_reward = 0

        while not board.is_done:
            state = board.agent_pos
            action = agent.choose_action(state, board.allowed_moves[state])
            new_state, reward = board.step(action)
            next_allowed = board.allowed_moves.get(new_state, [])
            agent.update(state, action, reward, new_state, next_allowed, board.is_done)
            total_reward += reward
        
        step_history.append(board.steps)
        reward_history.append(total_reward)
        board.reset()
        agent.reduce_exploration()
        
        if episode % 400 == 0:
            print(f"\033[4mEpisode {episode}\033[0m\nExploration rate: {agent.exploration_rate:1.3f}\n***")
    
    print("\n\033[93mLearnt moves\033[0m, \033[37minitial randomized moves\033[0m (highest Q-values):")
    to_arrows = {"U": "↑", "D": "↓", "L": "←", "R": "→"}
    print("╭───┬" + "───┬" * (GRID_SIZE - 2) + "───╮")
    for y in range(GRID_SIZE):
        row = []
        for x in range(GRID_SIZE):
            if (y, x) not in board.allowed_moves:
                row.append(" ▉ ")
            elif (y, x) in TRAPS:
                row.append(" \033[91m╳\033[0m ")
            elif (y, x) == GOAL:
                row.append(" \033[92m◯\033[0m ")
            else:
                allowed = board.allowed_moves[(y, x)]
                best_a = to_arrows[max(allowed, key = lambda a: agent.Q[((y, x), a)])]
                initial_best_a = to_arrows[max(allowed, key = lambda a: initial_Q[((y, x), a)])]
                row.append(f"\033[93m{best_a}\033[0m \033[37m{initial_best_a}\033[0m")
        print("│" + "│".join(row) + "│")
        if not y == GRID_SIZE - 1:
            print("├───┼" + "───┼" * (GRID_SIZE - 2) + "───┤")
        else:
            print("╰───┴" + "───┴" * (GRID_SIZE - 2) + "───╯")

    window = 50
    fig, axes = plt.subplots(1, 2, figsize = (12, 4))
    fig.suptitle("Treasure hunt metrics", fontsize = 14)

    axes[0].semilogy(step_history, "b-", alpha = 0.4, linewidth = 0.8)
    rolling = np.convolve(step_history, np.ones(window) / window, mode = "valid")
    axes[0].semilogy(range(window - 1, len(step_history)), rolling, "r-", linewidth = 2)
    axes[0].set_title("Steps per episode")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Steps (log scale)")
    axes[0].legend(["Steps", f"Mean across last {window} episodes"])

    axes[1].plot(reward_history, "g-", alpha = 0.4, linewidth = 0.8)
    rolling_r = np.convolve(reward_history, np.ones(window) / window, mode = "valid")
    axes[1].plot(range(window - 1, len(reward_history)), rolling_r, "r-", linewidth = 2)
    axes[1].set_title("Total reward per episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Total reward")
    axes[1].legend(["Reward", f"Mean across last {window} episodes"])

    plt.tight_layout()
    plt.savefig("treasure_hunt_RL_fig.png", dpi = 120)
    plt.show()
