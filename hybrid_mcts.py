import copy
import random
from candy_simulation import apply_move, merge_jelly_to_grid, extract_jelly_grid
from candy_simulation import ObjectivesTracker
from optimal_move_selection import evaluate_board
from candy_simulation import find_possible_moves


class MCTSNode:
    def __init__(self, candy_grid, jelly_grid, tracker, depth, move=None):
        self.candy_grid = candy_grid
        self.jelly_grid = jelly_grid
        self.tracker = tracker
        self.depth = depth
        self.move = move
        self.children = []
        self.visits = 0
        self.total_score = 0

    def is_terminal(self, max_depth):
        return self.depth >= max_depth

    def expand(self, possible_moves):
        for move in possible_moves:
            new_candy_grid = copy.deepcopy(self.candy_grid)
            new_jelly_grid = copy.deepcopy(self.jelly_grid)
            new_tracker = copy.deepcopy(self.tracker)

            grid_after, jelly_after = apply_move(new_candy_grid, new_jelly_grid,
                                                 move[0][0], move[0][1], move[1][0], move[1][1],
                                                 tracker=new_tracker)

            new_node = MCTSNode(grid_after, jelly_after, new_tracker, self.depth + 1, move)
            self.children.append(new_node)

    def simulate(self, objective_targets):
        # Perform greedy rollout with urgency weighting
        temp_tracker = copy.deepcopy(self.tracker)
        score = evaluate_board(temp_tracker.get_summary(), objective_targets)
        return score

    def backpropagate(self, score):
        self.visits += 1
        self.total_score += score


def prune_moves_with_heuristics(moves, grid, jelly_grid, objectives):
    """
    Removes low-value moves. Keep moves:
    - Near objective tiles
    - Affecting special candies
    - In top 80% of estimated value
    """
    def heuristic_value(r,c):
        label = grid[r][c]
        label = label[1] if isinstance(label, tuple) else label
        label_suffix = label.split('_')
        if label_suffix[-1] in ["H", "V", "W", "F"]:
            label_suffix = label_suffix[-1]
        else:
            label_suffix = ""
        candy_label = jelly_grid[r][c]
        val = 0
        for obj, urgency in objectives.items():
            if label in obj:
                val += 5
        if label_suffix in ["H", "V", "W", "F"]:
            if obj == "striped" and (label_suffix == "H" or label_suffix == "V"):
                val += 5
            if obj == "bomb" and label_suffix == "W":
                val += 5
            if obj == "fish" and label_suffix == "F":
                val += 5
        if candy_label > 0:
            val += 3
            
        if any(s in label for s in ["H", "V", "W", "F"]):
            val += 3
        if "bomb" in label:
            val += 5    
        return val

    scored = []
    for move in moves:
        (r1,c1), (r2,c2), l1, l2 = move
        score = heuristic_value(r1,c1) + heuristic_value(r2,c2)
        scored.append((score, move))
    scored.sort(reverse=True)
    top_moves = [m for score, m in scored if score > 0]
    if not top_moves:
        return [m for _, m in scored[:max(1, len(moves) // 4)]]
    return top_moves


def hybrid_mcts(grid, jelly_grid, moves, objective_targets, max_depth=2, simulations_per_move=5):
    best_move = None
    best_score = float('-inf')
    best_tracker_summary = None
    move_labels = [(a, b, grid[a[0]][a[1]], grid[b[0]][b[1]]) for a, b, _, _ in moves]
    pruned_moves = prune_moves_with_heuristics(move_labels, grid, jelly_grid,  objective_targets)

    for move in pruned_moves:
        (r1, c1), (r2, c2), _, _ = move
        temp_grid = copy.deepcopy(grid)
        temp_jelly = copy.deepcopy(jelly_grid)
        temp_tracker = ObjectivesTracker()

        grid_after, jelly_after = apply_move(temp_grid, temp_jelly, r1, c1, r2, c2, tracker=temp_tracker)
        node = MCTSNode(grid_after, jelly_after, temp_tracker, depth=max_depth, move=((r1, c1), (r2, c2)))

        for _ in range(simulations_per_move):
            expanded_moves = find_possible_moves(node.candy_grid)
            node.expand(expanded_moves)
            for child in node.children:
                score = child.simulate(objective_targets)
                child.backpropagate(score)
                node.backpropagate(score)

        avg_score = node.total_score / max(1, node.visits)
        if avg_score > best_score:
            best_score = avg_score
            best_move = ((r1, c1), (r2, c2))
            best_tracker = temp_tracker.get_summary()

    return best_move, best_score, best_tracker
