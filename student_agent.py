# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math


class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)


# def evaluate_board(board):
#     board = np.array(board)

#     # Base reward: more empty tiles = more room to play
#     empty_tiles = np.count_nonzero(board == 0)

#     return empty_tiles

#     base_score = 1 * empty_tiles

#     # Future move simulation
#     env = Game2048Env()
#     env.board = board.copy()
#     N = 10  # number of simulations
#     scoring_moves = 0
#     total_legal_moves = 0

#     for _ in range(N):
#         temp_env = copy.deepcopy(env)
#         prev_score = temp_env.score
#         steps = 0
#         while not temp_env.is_game_over() and steps < 3:
#             legal_actions = [a for a in range(4) if temp_env.is_move_legal(a)]
#             total_legal_moves += len(legal_actions)
#             if not legal_actions:
#                 break
#             action = random.choice(legal_actions)
#             temp_env.step(action)
#             steps += 1
#         if temp_env.score > prev_score:
#             scoring_moves += 1

#     # Combine evaluation
#     reward = (
#         100 * scoring_moves     # major reward for score-making simulations
#         + 0.5 * total_legal_moves # small reward for having more options
#     )
#     return reward

class NTupleEvaluator:
    def __init__(self):
        self.lut = defaultdict(float)
        self.tuples = self._generate_tuples()

    def _generate_tuples(self):
        # Select fixed patterns: rows and columns
        patterns = []
        for i in range(4):
            # Horizontal rows
            patterns.append([(i, 0), (i, 1), (i, 2), (i, 3)])
            # Vertical columns
            patterns.append([(0, i), (1, i), (2, i), (3, i)])
        # Optional: diagonals
        patterns.append([(0, 0), (1, 1), (2, 2), (3, 3)])
        patterns.append([(0, 3), (1, 2), (2, 1), (3, 0)])
        return patterns

    def board_to_tuple_index(self, board, positions):
        # Convert tile values to log2 to reduce LUT size
        index = []
        for (i, j) in positions:
            val = board[i, j]
            index.append(int(math.log2(val)) if val != 0 else 0)
        return tuple(index)

    def evaluate(self, board):
        total = 0.0
        for pattern in self.tuples:
            idx = self.board_to_tuple_index(board, pattern)
            total += self.lut[idx]
        return total

    def update(self, board, delta):
        # Used in training: update values in LUT
        for pattern in self.tuples:
            idx = self.board_to_tuple_index(board, pattern)
            self.lut[idx] += delta

# Instantiate once globally or pass as parameter
evaluator = NTupleEvaluator()

# Replace your evaluate_board function with this:
def evaluate_board(board):
    return evaluator.evaluate(np.array(board))



import copy
import math
import random
from collections import defaultdict

class MCTSNode:
    def __init__(self, env, parent=None, action=None):
        self.env = env
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_puct=1.0):
        best_score = -float("inf")
        best_node = None
        for action, child in self.children.items():
            if child.visits == 0:
                ucb_score = float("inf")
            else:
                exploit = child.value_sum / child.visits
                explore = c_puct * math.sqrt(math.log(self.visits + 1) / child.visits)
                ucb_score = exploit + explore
            if ucb_score > best_score:
                best_score = ucb_score
                best_node = child
        return best_node

    def expand(self):
        action = self.untried_actions.pop()
        new_env = copy.deepcopy(self.env)
        new_env.step(action)
        child_node = MCTSNode(new_env, parent=self, action=action)
        self.children[action] = child_node
        return child_node

    def backpropagate(self, value):
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backpropagate(value)

    def is_terminal_node(self):
        return self.env.is_game_over()


def mcts_search(root_env, num_simulations=100):
    root_node = MCTSNode(copy.deepcopy(root_env))

    for _ in range(num_simulations):
        node = root_node

        # Selection
        while not node.is_terminal_node() and node.is_fully_expanded():
            node = node.best_child()

        # Expansion
        if not node.is_terminal_node() and not node.is_fully_expanded():
            node = node.expand()

        # Simulation
        simulation_result = evaluate_board(node.env.board)

        # Backpropagation
        node.backpropagate(simulation_result)

    # Choose the action with the most visits
    best_action = max(root_node.children.items(), key=lambda item: item[1].visits)[0]
    return best_action


def get_action(state, score):
    env = Game2048Env()
    env.board = state.copy()
    return mcts_search(env, simulations=100, max_depth=5)






