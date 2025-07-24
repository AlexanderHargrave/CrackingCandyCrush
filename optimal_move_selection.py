import random
import copy
from candy_simulation import ObjectivesTracker
from candy_simulation import find_possible_moves, apply_move, merge_jelly_to_grid, reshuffle_candies

def evaluate_board(tracker_summary, objective_targets):
    """
    Weighted evaluation score for the board based on objectives, urgency, and indirect blocker progress.
    """
    score = 0

    for obj, required in objective_targets.items():
        progress = tracker_summary.get(obj, 0)
        remaining = max(required - progress, 0)
        urgency_weight = 1 + (1 / (remaining + 1))
        score += progress * urgency_weight * 10  

    blocker_weights = {
        "frosting": 0.1,
        "liquorice": 0.1,
        "bubblegum": 0.1,
    }

    for blocker, weight in blocker_weights.items():
        if blocker not in objective_targets:
            cleared = tracker_summary.get(blocker, 0)
            score += cleared * weight
    if "dragonegg" in objective_targets:
        score += tracker_summary.get("dragonegg_movement", 0) * 2  
        score += tracker_summary.get("dragonegg", 0) * 10
    if "glass" in objective_targets:
        score += tracker_summary.get("glass", 0) * 2 

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
    return total_score / simulations, tracker_summary


def depth_based_simulation(grid, jelly_grid, objective_targets, depth=2):
    """
    Recursive depth-limited simulation to evaluate best move using average score.
    """
    def simulate_recursive(g, j, depth_left, total_depth):
        if depth_left == 0:
            return 0
        best_score = -float('inf')
        moves = find_possible_moves(g)
        shuffle_count = 0
        while not moves:
            if shuffle_count > 3:
                return -float('inf')
            g = reshuffle_candies(g)
            shuffle_count += 1
            moves = find_possible_moves(g)
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
    best_tracker_summary = None
    shuffle_count = 0
    moves = find_possible_moves(grid)
    while not moves:
        if shuffle_count > 3:
            print("No more moves available after shuffling.")
            return None, best_score, best_tracker_summary
        grid = reshuffle_candies(grid)
        shuffle_count += 1
        moves = find_possible_moves(grid)
    for move in find_possible_moves(grid):
        score, tracker_summary = monte_carlo_score(grid, jelly_grid, move, objective_targets, simulations=simulations_per_move)
        if score > best_score:
            best_score = score
            best_move = move[:2]
            best_tracker_summary = tracker_summary
    return best_move, best_score, best_tracker_summary
def simulate_to_completion(candy_grid, jelly_grid, objective_targets, strategy_fn, max_steps=30, **strategy_kwargs):
    """
    Simulate a level by selecting the best move with a given strategy until objectives are complete or max steps reached.
    
    Args:
        candy_grid: Initial candy grid.
        jelly_grid: Initial jelly grid.
        objective_targets: Dict of objectives and their required counts.
        strategy_fn: A function that returns (best_move, score, tracker_summary)
        max_steps: Max number of moves allowed before stopping.
        **strategy_kwargs: Additional arguments passed to strategy_fn.

    Returns:
        steps_taken, final_tracker_summary, success (bool)
    """
    current_grid = copy.deepcopy(candy_grid)
    current_jelly = copy.deepcopy(jelly_grid)
    tracker = ObjectivesTracker()
    steps_taken = 0
    
    while steps_taken < max_steps:
        consecutive_shuffles = 0
        possible_moves = find_possible_moves(current_grid)
        while len(possible_moves) == 0:
            if consecutive_shuffles > 5:
                print("No more moves")
                break
            current_grid = reshuffle_candies(current_grid)
            consecutive_shuffles += 1
            
            possible_moves = find_possible_moves(current_grid)

        move, score, tracker_summary = strategy_fn(current_grid, current_jelly, objective_targets, **strategy_kwargs)
        
        if not move:
            print("Strategy failed to find a move.")
            break

        r1, c1 = move[0]
        r2, c2 = move[1]
        current_grid, current_jelly = apply_move(current_grid, current_jelly, r1, c1, r2, c2, tracker=tracker)
        steps_taken += 1

        # Check completion
        complete = True
        for obj, required in objective_targets.items():
            progress = tracker.get_summary().get(obj, 0)
            if progress < required:
                complete = False
                break

        if complete:
            return steps_taken, tracker.get_summary(), True, current_grid, current_jelly, score

    return steps_taken, tracker.get_summary(), False, current_grid, current_jelly, score