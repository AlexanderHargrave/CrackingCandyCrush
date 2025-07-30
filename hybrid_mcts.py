import copy
import random
import math
from candy_simulation import apply_move, merge_jelly_to_grid, extract_jelly_grid, find_possible_moves
from candy_simulation import ObjectivesTracker
from optimal_move_selection import evaluate_board
from candy_simulation import find_possible_moves


class MCTSNode:
    def __init__(self, candy_grid, jelly_grid, tracker, depth, move=None, parent=None):
        self.candy_grid = candy_grid
        self.jelly_grid = jelly_grid
        self.tracker = tracker
        self.depth = depth
        self.move = move
        self.parent = parent
        self.children = []

        self.visits = 0
        self.total_score = 0

    def is_terminal(self, max_depth):
        return self.depth >= max_depth

    def expand(self, possible_moves):
        top_moves = self.get_top_moves(possible_moves, top_k=4)
        for move in top_moves:
            new_candy_grid = copy.deepcopy(self.candy_grid)
            new_jelly_grid = copy.deepcopy(self.jelly_grid)
            new_tracker = copy.deepcopy(self.tracker)

            grid_after, jelly_after = apply_move(new_candy_grid, new_jelly_grid,
                                                 move[0][0], move[0][1], move[1][0], move[1][1],
                                                 tracker=new_tracker)

            new_node = MCTSNode(grid_after, jelly_after, new_tracker,
                                self.depth + 1, move=move, parent=self)
            self.children.append(new_node)

    def simulate(self, objective_targets, rollout_depth=3):
        temp_grid = copy.deepcopy(self.candy_grid)
        temp_jelly = copy.deepcopy(self.jelly_grid)
        temp_tracker = copy.deepcopy(self.tracker)

        for _ in range(rollout_depth):
            moves = find_possible_moves(temp_grid)
            if not moves:
                break
            move = random.choice(self.get_top_moves(moves, top_k=4))
            (r1, c1), (r2, c2), *_ = move
            temp_grid, temp_jelly = apply_move(temp_grid, temp_jelly, r1, c1, r2, c2, tracker=temp_tracker)

        return evaluate_board(temp_tracker.get_summary(), objective_targets)

    def backpropagate(self, score):
        self.visits += 1
        self.total_score += score
        if self.parent:
            self.parent.backpropagate(score)

    def best_child(self, c=1.41):
        return max(
            self.children,
            key=lambda child: (child.total_score / (child.visits + 1e-5)) +
                              c * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-5))
        )

    def get_top_moves(self, moves, top_k=4):
        def heuristic_value(r, c):
            label = self.candy_grid[r][c]
            jelly = self.jelly_grid[r][c]
            base = label[1] if isinstance(label, tuple) else label
            base_type = base.split('_')[0]
            suffix = base.split('_')[-1] if '_' in base else ''
            score = 0
            if jelly > 0:
                score += 3
            if suffix in ["H", "V", "W", "F"]:
                score += 2
            if "bomb" in base:
                score += 5
            return score

        scored = []
        for move in moves:
            (r1, c1), (r2, c2), _, _ = move
            score = heuristic_value(r1, c1) + heuristic_value(r2, c2)
            scored.append((score, move))

        scored.sort(reverse=True)
        return [m for _, m in scored[:top_k]]


def prune_moves_with_heuristics(moves, grid, jelly_grid, objectives):
    def heuristic_value(r, c):
        label = grid[r][c]
        jelly = jelly_grid[r][c]
        base = label[1] if isinstance(label, tuple) else label
        base_type = base.split('_')[0]
        suffix = base.split('_')[-1] if '_' in base else ''
        cardinal_directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        score = 0

        for obj in objectives:
            if obj in base or obj in base_type:
                score += 3
        for dr, dc in cardinal_directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                neighbor_label = grid[nr][nc]
                neighbor_base = neighbor_label[1] if isinstance(neighbor_label, tuple) else neighbor_label
                if "frosting" in neighbor_base:
                    score += 3
                elif "liquorice" in neighbor_base:
                    score += 2
                elif "bubblegum" in neighbor_base:
                    score += 3
                if dr == 1 and "dragonegg" in neighbor_base:
                    score += 4

        if jelly > 0 and "glass" in objectives:
            score += 8

        if suffix in ["H", "V", "W", "F"]:
            score += 0.1
        if "striped" in objectives:
            if suffix in ["H", "V"]:
                score += 2
        if "bomb" in objectives:
            if suffix in ["W"]:
                score += 2
        if "fish" in objectives:
            if suffix in ["F"]:
                score += 2
        if "bomb" in base:
            score += 5
        return score

    scored = []
    for move in moves:
        (r1, c1), (r2, c2), l1, l2 = move
        score = heuristic_value(r1, c1) + heuristic_value(r2, c2)
        scored.append((score, move))
    scored.sort(reverse=True)

    return [m for score, m in scored[:5]]


def hybrid_mcts(grid, jelly_grid, objective_targets, max_depth=3, simulations_per_move=3):
    best_move = None
    best_score = float('-inf')
    best_tracker = None

    moves = find_possible_moves(grid)
    shuffle_count = 0
    
    move_labels = [(a, b, grid[a[0]][a[1]], grid[b[0]][b[1]]) for a, b, _, _ in moves]
    pruned_moves = prune_moves_with_heuristics(move_labels, grid, jelly_grid, objective_targets)

    for move in pruned_moves:
        (r1, c1), (r2, c2), _, _ = move
        temp_grid = copy.deepcopy(grid)
        temp_jelly = copy.deepcopy(jelly_grid)
        temp_tracker = ObjectivesTracker()

        grid_after, jelly_after = apply_move(temp_grid, temp_jelly, r1, c1, r2, c2, tracker=temp_tracker)
        root = MCTSNode(grid_after, jelly_after, temp_tracker, depth=0, move=((r1, c1), (r2, c2)))

        for _ in range(simulations_per_move):
            node = root
            # Selection
            while node.children:
                node = node.best_child()

            # Expansion
            if not node.is_terminal(max_depth):
                possible = find_possible_moves(node.candy_grid)
                if possible:
                    node.expand(possible)
                    node = random.choice(node.children)

            # Simulation and backpropagation
            score = node.simulate(objective_targets)
            node.backpropagate(score)

        avg_score = root.total_score / max(1, root.visits)
        if avg_score > best_score:
            best_score = avg_score
            best_move = ((r1, c1), (r2, c2))
            best_tracker = temp_tracker.get_summary()

    return best_move, best_score, best_tracker
