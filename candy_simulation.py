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
def is_liquorice(label):
    if isinstance(label, tuple):
        label = label[1]
    return "liquorice" in label.lower()

def is_marmalade(label):
    if isinstance(label, tuple):
        label = label[1]
    return "marmalade" in label.lower()

def is_lock(label):
    if isinstance(label, tuple):
        label = label[1]
    return "lock" in label.lower()

def is_frosting(label):
    if isinstance(label, tuple):
        label = label[1]
    return "frosting" in label.lower()

def is_bubblegum(label):
    if isinstance(label, tuple):
        label = label[1]
    return "bubblegum" in label.lower()

def reduce_layer(label, base_name):
    """
    Generic reducer for frostingX / bubblegumX.
    Returns reduced label (or 'empty' if level 1).
    """
    import re
    match = re.match(f"{base_name}(\\d)", label)
    if not match:
        return label
    level = int(match.group(1))
    if level <= 1:
        return 'empty'
    return f"{base_name}{level - 1}"
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
    Enhanced clearing function with recursive cascading:
    - Handles jelly reduction
    - Removes marmalade/lock wrapping
    - Pops liquorice/marmalade neighbors
    - Reduces frosting/bubblegum layers
    - Bubblegum1 explodes 3x3 area
    - Special candies clear accordingly (striped, wrapped)
    - Color bomb clears the most common candy if cleared passively
    """
    def is_matchable(r, c):
        label = candy_grid[r][c]
        return is_movable(label) and not is_chocolate(label) and not is_non_interactive(label)
    rows, cols = len(candy_grid), len(candy_grid[0])

    # Directions for 4-neighbors and 3x3 area
    cardinal = [(-1,0), (1,0), (0,-1), (0,1)]
    surrounding = [(-1,-1), (-1,0), (-1,1),
                   (0,-1),  (0,0),  (0,1),
                   (1,-1),  (1,0),  (1,1)]

    # Normalize label helper
    def base_label(tile):
        return tile[1] if isinstance(tile, tuple) else tile

    # Recursive processing of clearing
    to_process = set(matched_positions)
    processed = set()

    while to_process:
        r, c = to_process.pop()
        if (r, c) in processed:
            continue
        processed.add((r, c))

        label = candy_grid[r][c]
        base = base_label(label)

        # Special candy clearing logic
        if is_special_candy(base):
            direction = base.split('_')[0][-1]  # e.g. 'H', 'V', 'W'
            if direction == 'H':
                for cc in range(cols):
                    if (r, cc) not in processed:
                        to_process.add((r, cc))
            elif direction == 'V':
                for rr in range(rows):
                    if (rr, c) not in processed:
                        to_process.add((rr, c))
            elif direction == 'W':  # wrapped candy
                for dr, dc in surrounding:
                    wr, wc = r + dr, c + dc
                    if is_valid_position(candy_grid, wr, wc) and (wr, wc) not in processed:
                        to_process.add((wr, wc))

        # Bubblegum1 explosion triggers 3x3 clear around it
        if "bubblegum1" in base:
            for dr, dc in surrounding:
                nr, nc = r + dr, c + dc
                if is_valid_position(candy_grid, nr, nc) and (nr, nc) not in processed:
                    to_process.add((nr, nc))

        # Check neighbors for marmalade, liquorice, frosting, bubblegum to affect
        for dr, dc in cardinal:
            nr, nc = r + dr, c + dc
            if not is_valid_position(candy_grid, nr, nc):
                continue
            neighbor = candy_grid[nr][nc]
            neighbor_base = base_label(neighbor)

            if is_liquorice(neighbor_base) or is_marmalade(neighbor_base):
                if (nr, nc) not in processed:
                    to_process.add((nr, nc))

            elif "frosting" in neighbor_base:
                # reduce frosting by 1 layer
                new_label = reduce_layer(neighbor_base, "frosting")
                candy_grid[nr][nc] = new_label

            elif "bubblegum" in neighbor_base:
                if "bubblegum1" in neighbor_base:
                    # trigger 3x3 explosion around bubblegum1
                    for dr2, dc2 in surrounding:
                        br, bc = nr + dr2, nc + dc2
                        if is_valid_position(candy_grid, br, bc) and (br, bc) not in processed:
                            to_process.add((br, bc))
                # reduce bubblegum layer by 1
                new_label = reduce_layer(neighbor_base, "bubblegum")
                candy_grid[nr][nc] = new_label

        # Color bomb logic - clears all candies of most common color when cleared passively
        if "bomb" in base:
            # count candies currently on board
            candy_count = {}
            for i in range(rows):
                for j in range(cols):
                    tile = candy_grid[i][j]
                    tile_base = base_label(tile)
                    if is_matchable(i, j):
                        norm = normalize_candy_name(tile_base)
                        candy_count[norm] = candy_count.get(norm, 0) + 1
            if candy_count:
                most_common = max(candy_count, key=candy_count.get)
                for i in range(rows):
                    for j in range(cols):
                        tile = candy_grid[i][j]
                        tile_base = base_label(tile)
                        if normalize_candy_name(tile_base) == most_common and (i,j) not in processed:
                            to_process.add((i,j))

        # Clear or unwrap current tile
        box = None
        if isinstance(label, tuple):
            box, name = label
        else:
            name = label

        if "marmalade" in name:
            name = name.replace("_marmalade", "")
            candy_grid[r][c] = (box, name) if box else name
        elif "lock" in name:
            name = name.replace("_lock", "")
            candy_grid[r][c] = (box, name) if box else name
        else:
            candy_grid[r][c] = 'empty'

        reduce_jelly_at(jelly_grid, r, c)


