import random
import copy
from collections import Counter, defaultdict
from candy_simulation import ObjectivesTracker
from candy_simulation import find_possible_moves, apply_move, merge_jelly_to_grid, reshuffle_candies
import numpy as np
import heapq
def evaluate_board(tracker_summary, objective_targets):
    """
    Weighted evaluation score for the board based on objectives, urgency, and other blockers.
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
def simulate_to_completion_with_ensemble(candy_grid, jelly_grid, objective_targets, strategies, max_steps=30, **kwargs):
    """
    Simulate a level using ensemble strategy. Track which strategies agree with ensemble's chosen move (excluding ties).

    Args:
        candy_grid: Initial candy grid.
        jelly_grid: Initial jelly grid.
        objective_targets: Dict of objectives and their required counts.
        strategies: Dict of name -> strategy function.
        max_steps: Max number of moves.
        **kwargs: Extra args passed to strategy functions.

    Returns:
        steps_taken: Number of moves used.
        tracker_summary: Final summary of progress.
        success: Bool indicating if all objectives were met.
        current_grid: Final candy grid.
        current_jelly: Final jelly grid.
        final_score: Score from final move.
        strategy_agreements: Dict of strategy_name -> count of agreements with ensemble.
        total_agreeable_steps: How many times the ensemble move wasn't a tie (used as denominator).
    """
    current_grid = copy.deepcopy(candy_grid)
    current_jelly = copy.deepcopy(jelly_grid)
    tracker = ObjectivesTracker()
    steps_taken = 0
    strategy_agreements = defaultdict(int)
    total_agreeable_steps = 0
    final_score = -float("inf")  # Default if nothing played

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

        move, score, _, agreeing_strategies, is_tiebreak = ensemble_strategy(
            current_grid, current_jelly, objective_targets, strategies, **kwargs
        )

        if not move:
            print("Ensemble strategy failed to find a move.")
            break

        # Apply the move
        r1, c1 = move[0]
        r2, c2 = move[1]
        current_grid, current_jelly = apply_move(current_grid, current_jelly, r1, c1, r2, c2, tracker=tracker)
        steps_taken += 1
        final_score = score

        if not is_tiebreak:
            total_agreeable_steps += 1
            for strat in agreeing_strategies:
                strategy_agreements[strat] += 1

        complete = True
        for obj, required in objective_targets.items():
            if tracker.get_summary().get(obj, 0) < required:
                complete = False
                break

        if complete:
            return steps_taken, tracker.get_summary(), True, current_grid, current_jelly, final_score, strategy_agreements, total_agreeable_steps

    return steps_taken, tracker.get_summary(), False, current_grid, current_jelly, final_score, strategy_agreements, total_agreeable_steps

def expectimax(grid, jelly_grid, objective_targets, depth=2):
    def expectimax_rec(g, j, tracker, curr_depth, is_max_node):
        if curr_depth == 0:
            return evaluate_board(tracker.get_summary(), objective_targets)

        if is_max_node:
            max_score = float('-inf')
            moves = find_possible_moves(g)
            if not moves:
                return evaluate_board(tracker.get_summary(), objective_targets)
            for move in moves:
                (r1, c1), (r2, c2), *_ = move
                g_copy = copy.deepcopy(g)
                j_copy = copy.deepcopy(j)
                tracker_copy = copy.deepcopy(tracker)
                new_g, new_j = apply_move(g_copy, j_copy, r1, c1, r2, c2, tracker=tracker_copy)
                score = expectimax_rec(new_g, new_j, tracker_copy, curr_depth - 1, False)
                max_score = max(max_score, score)
            return max_score
        else:
            # Chance node: sample a few random outcomes and average them
            samples = 3
            total = 0
            for _ in range(samples):
                g_copy = copy.deepcopy(g)
                j_copy = copy.deepcopy(j)
                tracker_copy = copy.deepcopy(tracker)
                random_moves = find_possible_moves(g_copy)
                if not random_moves:
                    total += evaluate_board(tracker_copy.get_summary(), objective_targets)
                    continue
                move = random.choice(random_moves)
                (r1, c1), (r2, c2), *_ = move
                new_g, new_j = apply_move(g_copy, j_copy, r1, c1, r2, c2, tracker=tracker_copy)
                score = expectimax_rec(new_g, new_j, tracker_copy, curr_depth - 1, True)
                total += score
            return total / max(1, samples)

    # Main loop to select the best move
    best_move = None
    best_score = float('-inf')
    best_summary = {}
    possible_moves = find_possible_moves(grid)

    for move in possible_moves:
        (r1, c1), (r2, c2), *_ = move
        g_copy = copy.deepcopy(grid)
        j_copy = copy.deepcopy(jelly_grid)
        tracker = ObjectivesTracker()
        new_g, new_j = apply_move(g_copy, j_copy, r1, c1, r2, c2, tracker=tracker)
        score = expectimax_rec(new_g, new_j, tracker, depth - 1, False)
        if score > best_score:
            best_score = score
            best_move = ((r1, c1), (r2, c2))
            best_summary = tracker.get_summary()

    return best_move, best_score, best_summary

def heuristic_score(tracker, objective_targets):
    """
    Basic heuristic score that can be expanded.
    """
    summary = tracker.get_summary()
    return evaluate_board(summary, objective_targets)


def heuristics_softmax_best_move(grid, jelly_grid, objective_targets, temperature=1.0, top_k=3):
    """
    Chooses a move based on softmax over heuristic scores.
    
    Parameters:
        temperature (float): Controls exploration (higher is more random).
        top_k (int or None): If set, limits sampling to top-k scored moves.
    
    Returns:
        (move), score, summary
    """
    moves = find_possible_moves(grid)
    if not moves:
        return None, -float('inf'), {}

    scored_moves = []
    for move in moves:
        (r1, c1), (r2, c2), *_ = move
        g_copy = copy.deepcopy(grid)
        j_copy = copy.deepcopy(jelly_grid)
        tracker = ObjectivesTracker()
        apply_move(g_copy, j_copy, r1, c1, r2, c2, tracker=tracker)
        score = heuristic_score(tracker, objective_targets)
        scored_moves.append(((r1, c1), (r2, c2), score, tracker.get_summary()))

    # Extract raw scores
    raw_scores = np.array([s[2] for s in scored_moves], dtype=np.float64)
    
    # Softmax with numerical stability
    stabilized_scores = raw_scores - np.max(raw_scores)
    exp_scores = np.exp(stabilized_scores / max(temperature, 1e-8))
    probs = exp_scores / (np.sum(exp_scores) + 1e-8)

    # Apply top-k filtering if needed
    if top_k is not None and top_k < len(scored_moves):
        # Sort by score and select top-k
        top_indices = np.argsort(raw_scores)[-top_k:]
        top_probs = probs[top_indices]
        top_probs /= np.sum(top_probs)
        chosen_index = np.random.choice(top_indices, p=top_probs)
    else:
        chosen_index = np.random.choice(len(scored_moves), p=probs)

    chosen = scored_moves[chosen_index]
    return (chosen[0], chosen[1]), chosen[2], chosen[3]
    

def ensemble_strategy(current_grid, current_jelly, objective_targets, strategies, **kwargs):
    move_votes = []
    strategy_results = {}

    for name, fn in strategies.items():
        try:
            move, score, summary = fn(copy.deepcopy(current_grid), copy.deepcopy(current_jelly), objective_targets, **kwargs)
            if move:
                move_votes.append(move)
                strategy_results[name] = (move, score, summary)
        except Exception:
            continue

    if not move_votes:
        return None, -float("inf"), {}, [], False

    vote_counts = Counter(move_votes)
    max_votes = max(vote_counts.values())
    modal_moves = [move for move, count in vote_counts.items() if count == max_votes]
    chosen_move = random.choice(modal_moves)
    is_tiebreak = len(modal_moves) > 1 

    agreeing_strategies = [name for name, (move, _, _) in strategy_results.items() if move == chosen_move]
    score, summary = strategy_results[agreeing_strategies[0]][1:]

    return chosen_move, score, summary, agreeing_strategies, is_tiebreak