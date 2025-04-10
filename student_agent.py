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
import os 
from collections import defaultdict 


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

class NTupleLUT:
    def __init__(self, board_dim, tuple_patterns):
        self.board_dim = board_dim
        self.patterns = tuple_patterns
        self.lut_weights = [defaultdict(float) for _ in self.patterns]
        self.pattern_symmetries = [self._generate_symmetries(p) for p in self.patterns]

        try:
            self._load_weights()
            print("LUTs loaded from file.")
        except Exception:
            print("Failed to load LUTs. Initialized with empty weights.")

    def _generate_symmetries(self, pattern):
        """
        Generate all symmetric variants (rotations and flips) of the given pattern.
        """
        all_variants = set()
        coords = np.array(pattern)

        for _ in range(4):  # Four 90-degree rotations
            coords = np.array([(y, self.board_dim - 1 - x) for x, y in coords])
            all_variants.add(tuple(map(tuple, coords)))

        # Horizontal flip
        h_flip = tuple((x, self.board_dim - 1 - y) for x, y in pattern)
        all_variants.add(h_flip)

        # Vertical flip
        v_flip = tuple((self.board_dim - 1 - x, y) for x, y in pattern)
        all_variants.add(v_flip)

        # Combined transforms
        h_rot = [(x, self.board_dim - 1 - y) for x, y in pattern]
        coords = np.array([(y, self.board_dim - 1 - x) for x, y in h_rot])
        all_variants.add(tuple(map(tuple, coords)))

        v_rot = [(self.board_dim - 1 - x, y) for x, y in pattern]
        coords = np.array([(y, self.board_dim - 1 - x) for x, y in v_rot])
        all_variants.add(tuple(map(tuple, coords)))

        return list(all_variants)

    def _tile_index(self, value):
        """
        Maps a tile value to an index for lookup.
        """
        return 0 if value == 0 else int(math.log(value, 2))

    def _extract_feature(self, board, pattern_coords):
        """
        Generate feature vector from board using tile indices at given pattern.
        """
        return tuple(self._tile_index(board[y, x]) for x, y in pattern_coords)

    def evaluate(self, board):
        """
        Estimate value of a board state using n-tuple patterns and symmetries.
        """
        total = 0.0
        for i, sym_list in enumerate(self.pattern_symmetries):
            for sym in sym_list:
                feature = self._extract_feature(board, sym)
                total += self.lut_weights[i][feature]
        return total

    def update_weights(self, board, td_error, learning_rate):
        """
        Perform TD update for all symmetric features of current board.
        """
        for i, sym_list in enumerate(self.pattern_symmetries):
            for sym in sym_list:
                feature = self._extract_feature(board, sym)
                self.lut_weights[i][feature] += learning_rate * td_error

    def save_weights(self, filename="ntuple_luts.pkl"):
        try:
            serializable = [dict(weights) for weights in self.lut_weights]
            with open(filename, 'wb') as file:
                pickle.dump(serializable, file)
            print(f"Weights saved to {filename}")
        except Exception as err:
            print(f"Failed to save LUTs: {err}")

    def _load_weights(self, filename="ntuple_luts.pkl"):
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                loaded = pickle.load(file)
            self.lut_weights = []
            for weights in loaded:
                d = defaultdict(float)
                d.update(weights)
                self.lut_weights.append(d)
        else:
            raise FileNotFoundError

def best_action(env, approximator):
    possible_moves = [a for a in range(4) if env.is_move_legal(a)]
    if not possible_moves:
        return None, 0

    original_board = env.board.copy()
    original_score = env.score

    best_score = -float('inf')
    best_move = None

    for move in possible_moves:
        _, move_score, _, afterstate = env.step(move)
        score_delta = move_score - original_score
        value = score_delta + approximator.evaluate(afterstate)
        if value > best_score:
            best_score = value
            best_move = move

        env.board = original_board.copy()
        env.score = original_score

    return best_move, best_score


def expected_value(afterstate, approximator):
    """
    Simulate all possible tile insertions and compute expected value.
    Takes into account the number of empty tiles.
    """
    total_value = 0
    empty_count = 0
    simulator = Game2048Env()
    simulator.board = afterstate.copy()

    # Count empty tiles and calculate potential value from them
    for i in range(4):
        for j in range(4):
            if simulator.board[i][j] == 0:
                empty_count += 1

                # Try inserting 2 and calculate the resulting value
                simulator.board[i][j] = 2
                _, score_2 = best_action(simulator, approximator)
                total_value += 0.9 * score_2  # Weight 2 tiles more

                # Try inserting 4 and calculate the resulting value
                simulator.board[i][j] = 4
                _, score_4 = best_action(simulator, approximator)
                total_value += 0.1 * score_4  # Weight 4 tiles less

                # Reset the tile after simulation
                simulator.board[i][j] = 0

    # Return the average value adjusted for the number of empty tiles
    # Higher empty_count implies more room for future moves, so this is factored in
    return (total_value / empty_count) if empty_count else 0



def run_training(env, approximator, episodes=5):
    scores = []

    for ep in range(episodes):
        env.reset()
        done = False

        while not done:
            action = best_action(env, approximator)
            _, _, done, _ = env.step(action)

        print(f"Episode {ep + 1} finished. Score: {env.score}")
        scores.append(env.score)

    avg_score = sum(scores) / len(scores)
    print("Average Score:", avg_score)
    return scores


# Define board patterns
tuple_patterns = [
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (3, 0)),
    ((0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)),
    ((0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (3, 1)),
]

# Create LUT approximator
approximator = NTupleLUT(board_dim=4, tuple_patterns=tuple_patterns)


def get_action(board_state, current_score):
    env = Game2048Env()
    env.board = board_state
    env.score = current_score
    return best_action(env, approximator)
