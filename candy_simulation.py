# candy_simulation.py
import random

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
    Extracts jelly levels from the main grid and returns two grids
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
            jelly_level = 0
            if "_jelly2" in label:
                label = label.replace("_jelly2", "")
                jelly_level = 2
            elif "_jelly1" in label:
                label = label.replace("_jelly1", "")
                jelly_level = 1
            candy_grid[r][c] = (box, label) if box is not None else label
            jelly_grid[r][c] = jelly_level

    return candy_grid, jelly_grid
def reduce_jelly_at(jelly_grid, r, c):
    """
    Reduces jelly at (r, c) by 1 level (if it's > 0). Used for when it get destroyed in match.
    """
    if jelly_grid[r][c] > 0:
        jelly_grid[r][c] -= 1
    return jelly_grid
def merge_jelly_to_grid(candy_grid, jelly_grid):
    """
    Recombines jelly information back into the candy labels for later usage.
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
    Only includes movable, matchable candies (excludes frosting, bubblegum, etc.).
    This is done after a move to identify all current game state.
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

def update_label(cell, new_label):
    if isinstance(cell, tuple):
        return (cell[0], new_label)
    return new_label
def trigger_wrapped_explosions(grid, jelly_grid):
    rows, cols = len(grid), len(grid[0])
    explosion_triggered = False

    surrounding = [(-1,-1), (-1,0), (-1,1),
                   (0,-1),  (0,0),  (0,1),
                   (1,-1),  (1,0),  (1,1)]

    new_matches = set()

    for r in range(rows):
        for c in range(cols):
            label = get_label(grid[r][c])
            if label.startswith("explode_"):
                explosion_triggered = True
                color = label.replace("explode_", "")
                for dr, dc in surrounding:
                    nr, nc = r + dr, c + dc
                    if is_valid_position(grid, nr, nc):
                        new_matches.add((nr, nc))
                # Clear the explode tile itself
                grid[r][c] = update_label(grid[r][c], "empty")
                jelly_grid = reduce_jelly_at(jelly_grid, r, c)

    if new_matches:
        grid, jelly_grid = clear_matches(grid, jelly_grid, new_matches)

    return grid, jelly_grid, explosion_triggered
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

    def base_label(tile):
        return tile[1] if isinstance(tile, tuple) else tile

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
            direction = base.split('_')[0][-1] 
            if direction == 'H':
                for cc in range(cols):
                    if (r, cc) not in processed:
                        to_process.add((r, cc))
            elif direction == 'V':
                for rr in range(rows):
                    if (rr, c) not in processed:
                        to_process.add((rr, c))
            elif direction == 'W':  
                for dr, dc in surrounding:
                    wr, wc = r + dr, c + dc
                    if is_valid_position(candy_grid, wr, wc) and (wr, wc) not in processed:
                        to_process.add((wr, wc))
                candy_grid[r][c] = update_label(label, f"explode_{base}")
                jelly_grid = reduce_jelly_at(jelly_grid, r, c)

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
                new_label = reduce_layer(neighbor_base, "frosting")
                candy_grid[nr][nc] = update_label(candy_grid[nr][nc], new_label)

            elif "bubblegum" in neighbor_base:
                if "bubblegum1" in neighbor_base:
                    # trigger 3x3 explosion around bubblegum1
                    for dr2, dc2 in surrounding:
                        br, bc = nr + dr2, nc + dc2
                        if is_valid_position(candy_grid, br, bc) and (br, bc) not in processed:
                            to_process.add((br, bc))
                new_label = reduce_layer(neighbor_base, "bubblegum")
                candy_grid[nr][nc] = update_label(candy_grid[nr][nc], new_label)

        # Chocolate bomb logic - clears all candies of most common color when cleared passively
        if "bomb" in base:
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

        box = None
        if isinstance(label, tuple):
            box, name = label
        else:
            name = label

        if "marmalade" in name:
            name = name.replace("_marmalade", "")
            candy_grid[r][c] = update_label(candy_grid[r][c], name)
        elif "lock" in name:
            name = name.replace("_lock", "")
            candy_grid[r][c] = update_label(candy_grid[r][c], name)
        else:
            candy_grid[r][c] = update_label(candy_grid[r][c], 'empty')

        jelly_grid = reduce_jelly_at(jelly_grid, r, c)
    return candy_grid, jelly_grid

def get_label(cell):
    return cell[1] if isinstance(cell, tuple) else cell

def apply_gravity(grid):
    """
    Applies gravity from top to bottom. If a movable candy has an empty space below,
    move it downward by transferring its label. Boxes (if any) are preserved.
    """
    rows, cols = len(grid), len(grid[0])
    changed = True
    updated = False
    while changed:
        changed = False
        for r in range(rows - 1): 
            for c in range(cols):
                curr = grid[r][c]
                below = grid[r + 1][c]

                if not is_movable(curr):
                    continue

                below_label = get_label(below) if below is not None else "empty"
                if below is None or "empty" in below_label:
                    # Move label downward
                    grid[r + 1][c] = update_label(below if below else curr, get_label(curr))
                    grid[r][c] = update_label(curr, "empty")
                    changed = True
                    updated = True
    return grid, updated

def get_new_candy():
    return random.choice(["red", "blue", "green", "purple", "orange"])

def get_dispenser_candy(loader):
    """
    Simulates candy dispenser logic based on loader position.
    Since dispenser logic is level dependent, for the purposes of this project it operates randomly.
    """
    dispense_choice = ["red", "blue", "green", "purple", "orange"]
    if "bomb" in loader:
        dispense_choice.append("redW")
        dispense_choice.append("blueW")
        dispense_choice.append("greenW")
        dispense_choice.append("purpleW")
        dispense_choice.append("orangeW")
    if "liquorice" in loader:
        dispense_choice.append("liquorice_swirl")
    if "vertical" in loader:
        dispense_choice.append("redV")
        dispense_choice.append("blueV")
        dispense_choice.append("greenV")
        dispense_choice.append("purpleV")
        dispense_choice.append("orangeV")
    if "hoirzontal" in loader:
        dispense_choice.append("redH")
        dispense_choice.append("blueH")
        dispense_choice.append("greenH")
        dispense_choice.append("purpleH")
        dispense_choice.append("orangeH")
    if "chocolate" in loader:
        dispense_choice.append("bomb")
    if "fish" in loader:
        dispense_choice.append("redF")
        dispense_choice.append("blueF")
        dispense_choice.append("greenF")
        dispense_choice.append("purpleF")
        dispense_choice.append("orangeF")
    if "egg" in loader:
        dispense_choice.append("dragonegg")
    return random.choice(dispense_choice)

def is_empty(cell):
    return cell is None or get_label(cell) == "empty"

def generate_and_fall_candies(grid):
    rows, cols = len(grid), len(grid[0])
    changed = False

    for c in range(cols):
        for r in range(rows):
            cell = grid[r][c]
            if is_empty(cell):
                above = None if r == 0 else grid[r - 1][c]
                above_label = get_label(above) if above else "gap"

                if r == 0 or "gap" in above_label or "loader" in above_label:
                    # Generate new candy
                    if above_label == "loader":
                        new_candy = get_dispenser_candy(get_label(grid[r - 1][c]))
                    else:
                        new_candy = get_new_candy()
                    fall_r = r
                    while fall_r + 1 < rows and is_empty(grid[fall_r + 1][c]):
                        fall_r += 1

                    grid[fall_r][c] = update_label(grid[fall_r][c], new_candy )
                    changed = True
    return grid, changed

def fill_grid_until_stable(grid):
    changed = True
    updated = 0
    while changed:
        grid, changed = generate_and_fall_candies(grid)
        updated += 1
    if updated >= 2:
        return grid, True
    return grid, False

def generate_at_top(grid, col):
    rows = len(grid)
    for r in range(rows):
        if is_empty(grid[r][col]):
            above = None if r == 0 else grid[r - 1][col]
            above_label = get_label(above) if above else "gap"

            if r == 0 or "gap" in above_label or "loader" in above_label:
                if above_label == "loader":
                    new_candy = get_dispenser_candy(get_label(grid[r - 1][col]))
                else:
                    new_candy = get_new_candy()

                fall_r = r
                while fall_r + 1 < rows and is_empty(grid[fall_r + 1][col]):
                    fall_r += 1
                grid[fall_r][col] = update_label(grid[fall_r][col], new_candy)
            break

def apply_diagonal_gravity(grid):
    rows, cols = len(grid), len(grid[0])
    changed = False

    def fall_column(col, stop_row):
        # Pull candies down above stop_row
        for r in reversed(range(stop_row)):
            if is_movable(grid[r][col]) and is_empty(grid[r + 1][col]):
                grid[r + 1][col] = grid[r][col]
                grid[r][col] = update_label(grid[r][col], "empty")
                return True
        return False

    for r in reversed(range(rows - 1)):
        for c in range(cols):
            curr = grid[r][c]
            if not is_movable(curr):
                continue

            # Try down-left
            if c > 0 and is_empty(grid[r + 1][c - 1]):
                grid[r + 1][c - 1] = curr
                grid[r][c] = update_label(curr, "empty")
                fall_column(c, r)
                generate_at_top(grid, c)
                changed = True
                continue

            # Try down-right
            if c < cols - 1 and is_empty(grid[r + 1][c + 1]):
                grid[r + 1][c + 1] = curr
                grid[r][c] = update_label(curr, "empty")
                fall_column(c, r)
                generate_at_top(grid, c)
                changed = True

    return grid, changed

def update_board(grid, jelly_grid):
    while True:
        # 1. Apply gravity and fill until stable
        changed = True
        while changed:
            changed = False
            grid, changed1 = apply_gravity(grid)
            grid, filled1 = fill_grid_until_stable(grid)
            grid, diag1 = apply_diagonal_gravity(grid)
            grid, filled2 = fill_grid_until_stable(grid)
            changed = changed1 or filled1 or diag1 or filled2

        # 2. Trigger wrapped candy explosions if any
        grid, jelly_grid, triggered = trigger_wrapped_explosions(grid, jelly_grid)
        if triggered:
            continue 

        # 3. Check for new matches on the updated board
        matches = find_all_matches(grid)
        if matches:
            
            grid, jelly_grid = clear_matches(grid, jelly_grid, matches)
            continue  

        # 4. No more matches or explosions — board is stable
        break

    return grid, jelly_grid



