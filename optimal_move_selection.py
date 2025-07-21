import random
import copy
from candy_simulation import ObjectivesTracker
from candy_simulation import find_possible_moves, apply_move, merge_jelly_to_grid


def evaluate_board(tracker_summary, objective_targets):
    """
    Weighted evaluation score for the board based on objectives and urgency.
    """
    score = 0
    for obj, required in objective_targets.items():
        progress = tracker_summary.get(obj, 0)
        remaining = max(required - progress, 0)
        urgency_weight = 1 + (1 / (remaining + 1))
        score += progress * urgency_weight
    return score


def simulate_with_tracker(grid, jelly_grid, move, objective_targets):
    """
    Simulates a move with a fresh tracker and returns the resulting board state and tracker summary.
    """
    grid_copy = copy.deepcopy(grid)
    jelly_copy = copy.deepcopy(jelly_grid)
    tracker = ObjectivesTracker()
    move = move[:2]
    (r1, c1), (r2, c2) = move
    grid_copy, jelly_copy = apply_move(grid_copy, jelly_copy, r1, c1, r2, c2, tracker=tracker)
    grid_copy = merge_jelly_to_grid(grid_copy, jelly_copy)
    return grid_copy, jelly_copy, tracker.get_summary()


def monte_carlo_score(grid, jelly_grid, move, objective_targets, simulations=5):
    """
    Run multiple simulations for a move to estimate expected objective gain.
    """
    total_score = 0
    for _ in range(simulations):
        g_copy, j_copy, tracker_summary = simulate_with_tracker(grid, jelly_grid, move, objective_targets)
        total_score += evaluate_board(tracker_summary, objective_targets)
    return total_score / simulations


def depth_based_simulation(grid, jelly_grid, objective_targets, depth=2):
    """
    Recursive depth-limited simulation to evaluate best move using average score.
    """
    def simulate_recursive(g, j, depth_left, total_depth):
        if depth_left == 0:
            return 0
        best_score = -float('inf')
        for move in find_possible_moves(g):
            g_copy, j_copy, tracker_summary = simulate_with_tracker(g, j, move, objective_targets)
            immediate_score = evaluate_board(tracker_summary, objective_targets)
            future_score = simulate_recursive(g_copy, j_copy, depth_left - 1, total_depth)
            average_score = (immediate_score + future_score) / total_depth
            best_score = max(best_score, average_score)
        return best_score

    best_move = None
    best_score = -float('inf')
    best_tracker_summary = None
    for move in find_possible_moves(grid):
        g_copy, j_copy, tracker_summary = simulate_with_tracker(grid, jelly_grid, move, objective_targets)
        immediate_score = evaluate_board(tracker_summary, objective_targets)
        future_score = simulate_recursive(g_copy, j_copy, depth - 1, depth)
        total_avg_score = (immediate_score + future_score) / depth
        if total_avg_score > best_score:
            best_score = total_avg_score
            best_move = move[:2]
            best_tracker_summary = tracker_summary

    return best_move, best_score, best_tracker_summary


def monte_carlo_best_move(grid, jelly_grid, objective_targets, simulations_per_move=5):
    """
    Picks the move with the highest expected value over N simulations.
    """
    best_move = None
    best_score = -float('inf')
    for move in find_possible_moves(grid):
        score = monte_carlo_score(grid, jelly_grid, move, objective_targets, simulations=simulations_per_move)
        if score > best_score:
            best_score = score
            best_move = move
    return best_move, best_score