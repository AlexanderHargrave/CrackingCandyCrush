# candy_simulation.py

def normalize_candy_name(label):
    """
    Extracts the base candy color/type from a label or a (box, label) tuple.
    """
    if isinstance(label, tuple):
        label = label[1]
    base = label.split('_')[0]
    if base.endswith(('H', 'V', 'W', 'F')):
        return base[:-1]
    return base

def is_valid_position(grid, row, col):
    return 0 <= row < len(grid) and 0 <= col < len(grid[0])

def is_movable(label):
    """
    Returns True if the tile is a normal, swappable candy.
    Filters out frosting, marmalade, bubblegum, gap, loader, etc.
    """
    if isinstance(label, tuple):
        label = label[1]

    lowered = label.lower()
    if any(x in lowered for x in ["frosting", "marmalade", "gap", "loader", "bubblegum", "empty", "lock", "pinball"]):
        return False
    return True
def is_non_interactive(label):
    """
    Returns True if the candy cannot interact with chocolate bombs or specials.
    """
    if isinstance(label, tuple):
        label = label[1]
    lowered = label.lower().replace(" ", "_")
    return any(x in lowered for x in ["liquorice", "dragon_egg"])
def swap(grid, r1, c1, r2, c2):
    """Swap two elements in a copy of the grid and return it."""
    new_grid = [row.copy() for row in grid]
    new_grid[r1][c1], new_grid[r2][c2] = new_grid[r2][c2], new_grid[r1][c1]
    return new_grid

def has_match(grid, r, c):
    """
    Checks if there's a match at position (r, c).
    A match means 3 or more normalized candies in a row or column.
    Liquorice swirls do not count as matches. So do not dragon eggs
    """
    label = grid[r][c]
    if not is_movable(label):
        return False
    if isinstance(label, tuple):
        label = label[1]
    if any(x in label.lower() for x in ["liquorice", "dragonegg"]):
        return False 
    

    target = normalize_candy_name(label)

    # Horizontal match
    count = 1
    for dc in [-1, -2]:
        nc = c + dc
        if is_valid_position(grid, r, nc):
            neighbor = grid[r][nc]
            if normalize_candy_name(neighbor) == target:
                count += 1
            else:
                break
    for dc in [1, 2]:
        nc = c + dc
        if is_valid_position(grid, r, nc):
            neighbor = grid[r][nc]
            if normalize_candy_name(neighbor) == target:
                count += 1
            else:
                break
    if count >= 3:
        return True

    # Vertical match
    count = 1
    for dr in [-1, -2]:
        nr = r + dr
        if is_valid_position(grid, nr, c):
            neighbor = grid[nr][c]
            if normalize_candy_name(neighbor) == target:
                count += 1
            else:
                break
    for dr in [1, 2]:
        nr = r + dr
        if is_valid_position(grid, nr, c):
            neighbor = grid[nr][c]
            if normalize_candy_name(neighbor) == target:
                count += 1
            else:
                break
    return count >= 3


def is_special_candy(label):
    if isinstance(label, tuple):
        label = label[1]
    base = label.split('_')[0]
    return base[-1] in ['H', 'V', 'W', 'F'] if len(base) > 1 else False

def is_chocolate(label):
    if isinstance(label, tuple):
        label = label[1]
    if "bomb" in label:
        return True
    return False

def is_valid_special_swap(c1, c2):
    # Both must be movable
    if not is_movable(c1) or not is_movable(c2):
        return False
    if (is_chocolate(c1) and is_non_interactive(c2)) or (is_chocolate(c2) and is_non_interactive(c1)):
        return False
    if is_chocolate(c1) and is_movable(c2):
        return True
    if is_chocolate(c2) and is_movable(c1):
        return True

    if is_special_candy(c1) and is_special_candy(c2):
        return True

    return False

def find_possible_moves(grid):
    moves = []
    special_moves = []
    rows, cols = len(grid), len(grid[0])

    for r in range(rows):
        for c in range(cols):
            if not is_movable(grid[r][c]):
                continue

            # Swap with right neighbor
            if is_valid_position(grid, r, c + 1) and is_movable(grid[r][c + 1]):
                swapped = swap(grid, r, c, r, c + 1)
                if has_match(swapped, r, c) or has_match(swapped, r, c + 1):
                    moves.append(((r, c), (r, c + 1), grid[r][c], grid[r][c + 1]))
                elif is_valid_special_swap(grid[r][c], grid[r][c + 1]):
                    special_moves.append(((r, c), (r, c + 1), grid[r][c], grid[r][c + 1]))

            # Swap with bottom neighbor
            if is_valid_position(grid, r + 1, c) and is_movable(grid[r + 1][c]):
                swapped = swap(grid, r, c, r + 1, c)
                if has_match(swapped, r, c) or has_match(swapped, r + 1, c):
                    moves.append(((r, c), (r + 1, c), grid[r][c], grid[r + 1][c]))
                elif is_valid_special_swap(grid[r][c], grid[r + 1][c]):
                    special_moves.append(((r, c), (r + 1, c), grid[r][c], grid[r + 1][c]))

    return moves + special_moves
def extract_jelly_grid(grid):
    """
    Extracts jelly levels from the main grid and returns two grids:
    - candy_grid: stripped of jelly suffix
    - jelly_grid: stores 0 (no jelly), 1 (jelly1), or 2 (jelly2)
    """
    rows, cols = len(grid), len(grid[0])
    candy_grid = [[None for _ in range(cols)] for _ in range(rows)]
    jelly_grid = [[0 for _ in range(cols)] for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            cell = grid[r][c]
            if isinstance(cell, tuple):
                box, label = cell
            else:
                label = cell
                box = None

            # Parse jelly level
            jelly_level = 0
            if "_jelly2" in label:
                label = label.replace("_jelly2", "")
                jelly_level = 2
            elif "_jelly1" in label:
                label = label.replace("_jelly1", "")
                jelly_level = 1

            # Assign cleaned values
            candy_grid[r][c] = (box, label) if box is not None else label
            jelly_grid[r][c] = jelly_level

    return candy_grid, jelly_grid
def reduce_jelly_at(jelly_grid, r, c):
    """
    Reduces jelly at (r, c) by 1 level (if it's > 0).
    """
    if jelly_grid[r][c] > 0:
        jelly_grid[r][c] -= 1
def merge_jelly_to_grid(candy_grid, jelly_grid):
    """
    Recombines jelly information back into the candy labels for display or saving.
    """
    rows, cols = len(candy_grid), len(candy_grid[0])
    merged_grid = [[None for _ in range(cols)] for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            cell = candy_grid[r][c]
            jelly_level = jelly_grid[r][c]

            if isinstance(cell, tuple):
                box, label = cell
            else:
                box, label = None, cell

            if jelly_level == 1:
                label += "_jelly1"
            elif jelly_level == 2:
                label += "_jelly2"

            merged_grid[r][c] = (box, label) if box else label

    return merged_grid

def find_all_matches(candy_grid):
    """
    Finds all horizontal and vertical matches of 3 or more candies.
    Returns a set of (r, c) positions that are part of matches.
    Only includes movable, matchable candies (excludes frosting, chocolate, etc.).
    """
    rows, cols = len(candy_grid), len(candy_grid[0])
    matched = set()

    def get_label(r, c):
        label = candy_grid[r][c]
        return normalize_candy_name(label)

    def is_matchable(r, c):
        label = candy_grid[r][c]
        return is_movable(label) and not is_chocolate(label) and not is_non_interactive(label)

    # Horizontal matches
    for r in range(rows):
        c = 0
        while c < cols - 2:
            if not is_matchable(r, c):
                c += 1
                continue
            label = get_label(r, c)
            count = 1
            while c + count < cols and is_matchable(r, c + count) and get_label(r, c + count) == label:
                count += 1
            if count >= 3:
                for i in range(count):
                    if (r,c+i) not in matched:

                        matched.add((r, c + i))
            c += max(count, 1)

    # Vertical matches
    for c in range(cols):
        r = 0
        while r < rows - 2:
            if not is_matchable(r, c):
                r += 1
                continue
            label = get_label(r, c)
            count = 1
            while r + count < rows and is_matchable(r + count, c) and get_label(r + count, c) == label:
                count += 1
            if count >= 3:
                for i in range(count):
                    if (r+i, c) not in matched:
                        matched.add((r + i, c))
            r += max(count, 1)

    return matched
def clear_matches(candy_grid, jelly_grid, matched_positions):
    """
    Clears matched candies by setting them to 'empty'.
    Also reduces jelly level at those positions.
    """
    for r, c in matched_positions:
        candy_grid[r][c] = 'empty'
        reduce_jelly_at(jelly_grid, r, c)

