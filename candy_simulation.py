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

def swap(grid, r1, c1, r2, c2):
    """Swap two elements in a copy of the grid and return it."""
    new_grid = [row.copy() for row in grid]
    new_grid[r1][c1], new_grid[r2][c2] = new_grid[r2][c2], new_grid[r1][c1]
    return new_grid

def has_match(grid, r, c):
    """
    Checks if there's a match at position (r, c).
    A match means 3 or more normalized candies in a row or column.
    Liquorice swirls do not count as matches.
    """
    label = grid[r][c]
    if not is_movable(label):
        return False
    if isinstance(label, tuple):
        label = label[1]
    if "liquorice" in label.lower():
        return False  # can't form matches with liquorice

    target = normalize_candy_name(label)

    # Horizontal match
    count = 1
    for dc in [-1, -2]:
        nc = c + dc
        if is_valid_position(grid, r, nc):
            neighbor = grid[r][nc]
            if "liquorice" in str(neighbor).lower():
                break
            if normalize_candy_name(neighbor) == target:
                count += 1
            else:
                break
    for dc in [1, 2]:
        nc = c + dc
        if is_valid_position(grid, r, nc):
            neighbor = grid[r][nc]
            if "liquorice" in str(neighbor).lower():
                break
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
            if "liquorice" in str(neighbor).lower():
                break
            if normalize_candy_name(neighbor) == target:
                count += 1
            else:
                break
    for dr in [1, 2]:
        nr = r + dr
        if is_valid_position(grid, nr, c):
            neighbor = grid[nr][c]
            if "liquorice" in str(neighbor).lower():
                break
            if normalize_candy_name(neighbor) == target:
                count += 1
            else:
                break
    return count >= 3

def is_special_candy(label):
    """
    Checks if a candy is a special candy (H, V, W, F), regardless of color.
    """
    if isinstance(label, tuple):
        label = label[1]
    base = label.split('_')[0]
    return base[-1] in ['H', 'V', 'W', 'F'] if len(base) > 1 else False

def is_chocolate(label):
    """
    Checks if a tile is a chocolate.
    """
    if isinstance(label, tuple):
        label = label[1]
    return "chocolate" in label.lower()
def is_valid_special_swap(c1, c2):
    """
    Returns True if the pair is a valid special/chocolate swap.
    """
    if not is_movable(c1) or not is_movable(c2):
        return False

    if is_chocolate(c1) and is_movable(c2):
        return True
    if is_chocolate(c2) and is_movable(c1):
        return True

    if is_special_candy(c1) and is_special_candy(c2):
        return True

    return False
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

