# candy_simulation.py
import random
import re
from collections import defaultdict, deque
from copy import deepcopy
from random import shuffle
class ObjectivesTracker:
    def __init__(self):
        self.counts = {
            "blue": 0,
            "green": 0,
            "red": 0,
            "purple": 0, 
            "orange": 0,
            "bomb": 0,
            "striped": 0,
            "fish": 0,
            "bubblegum": 0,
            "dragonegg": 0,
            "frosting": 0,
            "glass": 0,
            "liquorice": 0,
            "gumball": 0,
            "dragonegg_movement":0,
        }

    def update_on_clear(self, cell):
        """
        Called whenever a cell is cleared from the board.
        `cell` is a tuple: (box, label)
        `neighbors` is a list of adjacent cell labels (optional, used for gumball)
        """
        if not cell:
            return

        _, label = cell if isinstance(cell, tuple) else (None, cell)
        base_label = label.split("_")[0]
        end_label = ""
        # get last letter of base_label
        if base_label.endswith(('H', 'V', 'W', 'F')):
            end_label = base_label[-1]
            base_label = base_label[:-1]
        # Special candy types
        if end_label in ["H", "V", "W", "F"]:
            if end_label == "H":
                self.counts["striped"] += 1
            elif end_label == "V":
                self.counts["striped"] += 1
            elif end_label == "W":
                self.counts["bomb"] += 1
            elif end_label == "F":
                self.counts["fish"] += 1
        if base_label == "blue":
            self.counts["blue"] += 1
        elif base_label == "green":
            self.counts["green"] += 1
        elif base_label == "red":
            self.counts["red"] += 1
        elif base_label == "purple":
            self.counts["purple"] += 1
        elif base_label == "orange":
            self.counts["orange"] += 1
        elif base_label == "liquorice":
            self.counts["liquorice"] += 1
    def pinball_destroyed(self):
        self.counts["gumball"] += 1

    def jelly_destroyed(self):
        self.counts["glass"] += 1
    def on_liquorice_swirl(self):
        self.counts["liquorice"] += 1
    def on_bubblegum_destroyed(self):
        self.counts["bubblegum"] += 1
    def on_frosting_destroyed(self):
        self.counts["frosting"] += 1


    def on_dragonegg_removed(self):
        self.counts["dragonegg"] += 1
    def dragon_egg_movement(self, distance_travelled):
        self.counts["dragonegg_movement"] += distance_travelled

    def get_summary(self):
        return dict(self.counts)
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

def swap(grid, r1, c1, r2, c2):
    """Swap two elements in a copy of the grid and return it."""
    new_grid = [row.copy() for row in grid]
    new_grid[r1][c1], new_grid[r2][c2] = new_grid[r2][c2], new_grid[r1][c1]
    return new_grid
def reduce_layer(label, base_name, tracker = None):
    """
    Returns reduced label (or 'empty' if level 1).
    """
    
    match = re.match(f"{base_name}(\\d)", label)
    if not match:
        return label
    level = int(match.group(1))
    if level <= 1:
        if tracker:
            if base_name == "frosting":
                tracker.on_frosting_destroyed()
            elif base_name == "bubblegum":
                tracker.on_bubblegum_destroyed()
        return 'empty'
    if tracker:
        if base_name == "frosting":
            tracker.on_frosting_destroyed()
        elif base_name == "bubblegum":
            tracker.on_bubblegum_destroyed()
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
def reduce_jelly_at(jelly_grid, r, c, tracker= None):
    """
    Reduces jelly at (r, c) by 1 level (if it's > 0). Used for when it get destroyed in match.
    """
    if jelly_grid[r][c] > 0:
        jelly_grid[r][c] -= 1
        if tracker:
            tracker.jelly_destroyed()
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

            merged_grid[r][c] = (box, label) if box is not None else label

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
def trigger_wrapped_explosions(grid, jelly_grid, tracker = None):
    rows, cols = len(grid), len(grid[0])
    explosion_triggered = False

    surrounding = [(-1,-1), (-1,0), (-1,1),
                   (0,-1),  (0,0),  (0,1),
                   (1,-1),  (1,0),  (1,1)]

    new_matches = set()
    for r in range(rows):
        for c in range(cols):
            label = get_label(grid[r][c])
            if "explode" in label:
                explosion_triggered = True
                for dr, dc in surrounding:
                    nr, nc = r + dr, c + dc
                    if is_valid_position(grid, nr, nc):
                        new_matches.add((nr, nc))
                # Clear the explode tile itself
                grid[r][c] = update_label(grid[r][c], "yellow")
                jelly_grid = reduce_jelly_at(jelly_grid, r, c, tracker)

    if new_matches:
        grid, jelly_grid = clear_matches(grid, jelly_grid, new_matches, tracker)

    return grid, jelly_grid, explosion_triggered
def group_connected_matches(candy_grid, matched_positions):
    """Groups matched positions into connected sets of the same base color."""
    visited = set()
    groups = []
    def base_label(tile):
        return tile[1] if isinstance(tile, tuple) else tile
    for pos in matched_positions:
        if pos in visited:
            continue
        r0, c0 = pos
        ref_color = normalize_candy_name(base_label(candy_grid[r0][c0]))

        group = set()
        queue = deque([pos])

        while queue:
            r, c = queue.popleft()
            if (r, c) in visited:
                continue
            current_color = normalize_candy_name(base_label(candy_grid[r][c]))
            if current_color != ref_color:
                continue
            visited.add((r, c))
            group.add((r, c))

            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in matched_positions and (nr, nc) not in visited:
                    queue.append((nr, nc))
        if group:
            groups.append(group)
    return groups
def is_L_or_T_shape(group):


    row_counts = defaultdict(int)
    col_counts = defaultdict(int)
    for r, c in group:
        row_counts[r] += 1
        col_counts[c] += 1

    # L or T shapes have:
    # - at least one row and one column with ≥3 tiles
    return (
        any(count >= 3 for count in row_counts.values()) and
        any(count >= 3 for count in col_counts.values()))
def detect_and_mark_special_candy(candy_grid, group):
    def base_label(tile):
        return tile[1] if isinstance(tile, tuple) else tile

    if len(group) < 4:
        return None

    group = list(group)
    rows = [r for r, _ in group]
    cols = [c for _, c in group]
    r0, c0 = group[0]
    color = normalize_candy_name(base_label(candy_grid[r0][c0]))
    if color == "empty":
        return None
    is_horiz = len(set(rows)) == 1
    is_vert = len(set(cols)) == 1

    # Check for straight lines
    if len(group) == 4 and is_horiz:
        special = f"{color}V"
    elif len(group) == 4 and is_vert:
        special = f"{color}H"
    elif len(group) == 4 and len(set(rows)) == 2 and len(set(cols)) == 2:
        special = f"{color}F"
    elif len(group) >= 5 and (is_horiz or is_vert):
        special = "bomb"
    elif is_L_or_T_shape(group):
        special = f"{color}W"
    else:
        return None

    # Place special candy at the center or arbitrary position
    r_sp, c_sp = sorted(group)[len(group)//2]
    candy_grid[r_sp][c_sp] = update_label(candy_grid[r_sp][c_sp], special)
    return (r_sp, c_sp)

def clear_matches(candy_grid, jelly_grid, matched_positions, tracker = None):
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
    def spawn_fish_target(processed, to_process):
        candidates = {'bubblegum': [], 'frosting': [], 'liquorice': [], 'jelly': [], 'fallback': []}
        for r in range(rows):
            for c in range(cols):
                if (r, c) in processed or (r, c) in to_process:
                    continue
                tile = candy_grid[r][c]
                base = base_label(tile)

                if "bubblegum" in base:
                    candidates['bubblegum'].append((r, c))
                elif "frosting" in base:
                    candidates['frosting'].append((r, c))
                elif is_liquorice(base):
                    candidates['liquorice'].append((r, c))
                elif jelly_grid[r][c] > 0:
                    candidates['jelly'].append((r, c))
                else:
                    candidates['fallback'].append((r, c))

        for key in ['bubblegum', 'frosting', 'liquorice', 'jelly', 'fallback']:
            if candidates[key]:
                return random.choice(candidates[key])
        return None
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
    match_groups = group_connected_matches(candy_grid, matched_positions)
    created_specials = []
    for group in match_groups:
        special_pos = detect_and_mark_special_candy(candy_grid, group)
        if special_pos:
            created_specials.append(special_pos)
    to_process = set(matched_positions) | set(created_specials)
    while to_process:
        r, c = to_process.pop()
        if (r, c) in processed:
            continue
        processed.add((r, c))

        label = candy_grid[r][c]
        base = base_label(label)
        if "empty" in base or base == "gap" or "loader" in base:
            if "empty" in base:
                jelly_grid = reduce_jelly_at(jelly_grid, r, c, tracker)
            continue
        if tracker:
            tracked_label = base.split('_')[0]
            tracker.update_on_clear(tracked_label)
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
            elif direction == 'F':
                target = spawn_fish_target(processed, to_process)
                if target:
                    to_process.add(target)
                

        # Bubblegum1 explosion triggers 3x3 clear around it
        if "bubblegum1" in base:
            for dr, dc in surrounding:
                nr, nc = r + dr, c + dc
                if is_valid_position(candy_grid, nr, nc) and (nr, nc) not in processed:
                    to_process.add((nr, nc))

        # Check neighbors for marmalade, liquorice, frosting, bubblegum to affect
        if not (is_marmalade(base) or "frosting" in base or "bubblegum" in base or is_liquorice(base) or "dragonegg" in base or "pinball" in base):
            for dr, dc in cardinal:
                nr, nc = r + dr, c + dc
                if not is_valid_position(candy_grid, nr, nc):
                    continue

                neighbor = candy_grid[nr][nc]
                neighbor_base = base_label(neighbor)

                if is_marmalade(neighbor_base) or is_liquorice(neighbor_base):
                    to_process.add((nr, nc))

                elif "frosting" in neighbor_base:
                    new_label = reduce_layer(neighbor_base, "frosting", tracker)
                    candy_grid[nr][nc] = update_label(candy_grid[nr][nc], new_label)

                elif "bubblegum" in neighbor_base:
                    if "bubblegum1" in neighbor_base:
                        # Only 3x3 explosion effect applies
                        for dr2, dc2 in surrounding:
                            br, bc = nr + dr2, nc + dc2
                            if is_valid_position(candy_grid, br, bc) and (br, bc) not in processed:
                                to_process.add((br, bc))
                    new_label = reduce_layer(neighbor_base, "bubblegum", tracker)
                    candy_grid[nr][nc] = update_label(candy_grid[nr][nc], new_label)
                elif "pinball" in neighbor_base:
                    if tracker:
                        tracker.pinball_destroyed()

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
        base = name.split('_')[0]
        direction = base[-1] if len(base) > 1 else None
        if "marmalade" in name:
            name = name.replace("_marmalade", "")
            candy_grid[r][c] = update_label(candy_grid[r][c], name)
        elif "lock" in name:
            name = name.replace("_lock", "")
            candy_grid[r][c] = update_label(candy_grid[r][c], name)
        elif direction == "W":
            candy_grid[r][c] = update_label(candy_grid[r][c], "explode")
        elif "dragonegg" in base:
            pass
        elif "frosting" in base:
            new_label = reduce_layer(base, "frosting", tracker)
            if new_label == "empty":
                candy_grid[r][c] = update_label(candy_grid[r][c], "empty")
            else:
                candy_grid[r][c] = update_label(candy_grid[r][c], new_label)
        elif "bubblegum" in base:
            new_label = reduce_layer(base, "bubblegum", tracker)
            if new_label == "empty":
                candy_grid[r][c] = update_label(candy_grid[r][c], "empty")
            else:
                candy_grid[r][c] = update_label(candy_grid[r][c], new_label)
        elif "pinball" in base:
            if tracker:
                tracker.pinball_destroyed()
        else:
            candy_grid[r][c] = update_label(candy_grid[r][c], 'empty')

        jelly_grid = reduce_jelly_at(jelly_grid, r, c, tracker)
    return candy_grid, jelly_grid

def get_label(cell):
    return cell[1] if isinstance(cell, tuple) else cell

def apply_gravity(grid, tracker = None):
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
                    if "dragonegg" in get_label(curr) and tracker:
                        tracker.dragon_egg_movement(1)

                    grid[r + 1][c] = update_label(below if below else curr, get_label(curr))
                    grid[r][c] = update_label(curr, "empty")
                    changed = True
                    updated = True
    return grid, updated

def get_new_candy():
    return ["red", "blue", "green", "purple", "orange"]

def get_dispenser_candy(loader):
    """
    Simulates candy dispenser logic based on loader position.
    Since dispenser logic is level dependent, for the purposes of this project it operates randomly.
    """
    dispense_choice = []
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
    if "horizontal" in loader:
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
    return dispense_choice
def is_empty(cell):
    return cell is None or get_label(cell) == "empty"

def triggers_special_candy(grid, matched_positions):
    groups = group_connected_matches(grid, matched_positions)
    for group in groups:
        rows = [r for r, _ in group]
        cols = [c for _, c in group]
        if len(group) >= 5:
            return True  # Bomb or wrapped
        if len(group) == 4 and (len(set(rows)) == 1 or len(set(cols)) == 1):
            return True  # Striped
        if is_L_or_T_shape(group):
            return True  # Wrapped
    return False
def fill_grid_until_stable(grid):
    changed = True
    updated = 0
    new_candies = set()

    while changed:
        rows, cols = len(grid), len(grid[0])
        changed = False
        for c in range(cols):
            for r in range(rows):
                if is_empty(grid[r][c]):
                    fall_r = r
                    while fall_r + 1 < rows and is_empty(grid[fall_r + 1][c]):
                        fall_r += 1
                    pos = (fall_r, c)
                    new_candies.add(pos)
                    above = None if r == 0 else grid[r - 1][c]
                    above_label = get_label(above) if above else "gap"
                    if "loader" in above_label:
                        if random.random() < 0.5:
                            color = random.choice(get_new_candy())
                        else:

                            color = random.choice(get_dispenser_candy(above_label))
                    else:
                        color = random.choice(get_new_candy())
                    grid[fall_r][c] = update_label(grid[fall_r][c], color)

                    changed = True
        updated += 1

    return grid, updated >= 2, new_candies
def scramble_until_no_specials(grid, new_candies):
    base_colors = get_new_candy()
    max_attempts = 5

    for _ in range(max_attempts):
        matches = find_all_matches(grid)
        if not triggers_special_candy(grid, matches):
            return grid  # Done

        # Check if any group triggering special includes newly generated tile
        groups = group_connected_matches(grid, matches)
        changed = False

        for group in groups:
            if triggers_special_candy(grid, group):
                if any(pos in new_candies for pos in group):
                    for r, c in group:
                        if (r, c) in new_candies:
                            grid[r][c] = update_label(grid[r][c], random.choice(base_colors))
                            changed = True
        if not changed:
            break  # Avoid infinite loop
    return grid

def apply_diagonal_gravity(grid, tracker = None):
    rows, cols = len(grid), len(grid[0])
    changed = False

    def fall_column(col, stop_row):
        # Pull candies down above stop_row
        for r in reversed(range(stop_row)):
            if is_movable(grid[r][col]) and is_empty(grid[r + 1][col]):
                if "dragonegg" in get_label(grid[r][col]) and tracker:
                    tracker.dragon_egg_movement(1)
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
                #generate_at_top(grid, c)
                changed = True
                continue

            # Try down-right
            if c < cols - 1 and is_empty(grid[r + 1][c + 1]):
                grid[r + 1][c + 1] = curr
                grid[r][c] = update_label(curr, "empty")
                fall_column(c, r)
                #generate_at_top(grid, c)
                changed = True

    return grid, changed
# Write a function to check the grid for dragon egg and if it's on the bottom row or right below it is gap, clear it
def check_dragon_egg(grid, tracker = None):
    rows, cols = len(grid), len(grid[0])
    check = False
    for c in range(cols):
        for r in range(rows - 1, -1, -1):
            if "dragonegg" in get_label(grid[r][c]):
                if r == rows - 1 or (r < rows - 1 and get_label(grid[r + 1][c]) == "gap") and "marmalade" not in get_label(grid[r][c] and "lock" not in get_label(grid[r][c])):
                    if tracker:
                        tracker.on_dragonegg_removed()
                    check = True
                    grid[r][c] = update_label(grid[r][c], "empty")
    return grid, check
def update_board(grid, jelly_grid, tracker = None):
    while True:
        # 1. Apply gravity and fill until stable
        changed = True
        while changed:
            changed = False
            grid, changed1 = apply_gravity(grid, tracker)
            grid, filled1, new_candies = fill_grid_until_stable(grid)
            grid = scramble_until_no_specials(grid, new_candies)
            grid, diag1 = apply_diagonal_gravity(grid, tracker)
            grid, filled2, new_candies = fill_grid_until_stable(grid)
            grid = scramble_until_no_specials(grid, new_candies)
            changed = changed1 or filled1 or diag1 or filled2
        grid, check = check_dragon_egg(grid, tracker)
        if check:
            continue
        # 2. Trigger wrapped candy explosions if any
        grid, jelly_grid, triggered = trigger_wrapped_explosions(grid, jelly_grid, tracker)
        if triggered:
            #print("triggered wrapped explosion")
            continue 

        # 3. Check for new matches on the updated board
        matches = find_all_matches(grid)
        if matches:
            
            grid, jelly_grid = clear_matches(grid, jelly_grid, matches, tracker)
            continue  
        
        # 4. No more matches or explosions — board is stable
        break

    return grid, jelly_grid
def apply_swap(grid, jelly_grid, r1, c1, r2, c2):
    """
    Swaps two positions and handles special candy interactions.
    Returns:
        - updated grid after applying transformations
        - set of matched positions (for clear_matches)
    """
    rows, cols = len(grid), len(grid[0])
    new_grid = [row.copy() for row in grid]
    matched = set()

    def base_label(tile):
        return tile[1] if isinstance(tile, tuple) else tile

    def norm_name(tile):
        return normalize_candy_name(base_label(tile))

    def get_special_suffix(label):
        if isinstance(label, tuple):
            label = label[1]
        base = label.split('_')[0]
        if len(base) > 1 and base[-1] in ['H', 'V', 'W', 'F']:
            return base[-1]
        return None
    def spawn_fish_target(processed, to_process):
        candidates = {'bubblegum': [], 'frosting': [], 'liquorice': [], 'jelly': [], 'fallback': []}
        for r in range(rows):
            for c in range(cols):
                if (r, c) in processed or (r, c) in to_process:
                    continue
                tile = grid[r][c]
                base = base_label(tile)

                if "bubblegum" in base:
                    candidates['bubblegum'].append((r, c))
                elif "frosting" in base:
                    candidates['frosting'].append((r, c))
                elif is_liquorice(base):
                    candidates['liquorice'].append((r, c))
                elif jelly_grid[r][c] > 0:
                    candidates['jelly'].append((r, c))
                else:
                    candidates['fallback'].append((r, c))

        for key in ['bubblegum', 'frosting', 'liquorice', 'jelly', 'fallback']:
            if candidates[key]:
                return random.choice(candidates[key])
        return None
    # Perform the swap
    c1_label, c2_label = grid[r1][c1], grid[r2][c2]
    #new_grid[r1][c1], new_grid[r2][c2] = c2_label, c1_label
    new_grid[r1][c1], new_grid[r2][c2] = c2_label, c1_label
    c1_label = new_grid[r1][c1]
    c2_label = new_grid[r2][c2]
    is_choc1 = is_chocolate(c1_label)
    is_choc2 = is_chocolate(c2_label)
    is_spec1 = is_special_candy(base_label(c1_label))
    is_spec2 = is_special_candy(base_label(c2_label))

    # 🍫 Chocolate + Chocolate → clear entire board
    if is_choc1 and is_choc2:
        for r in range(rows):
            for c in range(cols):
                matched.add((r, c))
        new_grid[r1][c1] = update_label(new_grid[r1][c1], "empty")
        new_grid[r2][c2] = update_label(new_grid[r2][c2], "empty")

    elif (is_choc1 and is_spec2) or (is_choc2 and is_spec1):
        choc_r, choc_c = (r1, c1) if is_choc1 else (r2, c2)
        spec_tile = c2_label if is_choc1 else c1_label
        color = normalize_candy_name(spec_tile)
        suffix = get_special_suffix(spec_tile)

        if color and suffix:
            for r in range(rows):
                for c in range(cols):
                    if color in norm_name(new_grid[r][c]):
                        new_grid[r][c] = update_label(new_grid[r][c], f"{color}{suffix}")
        # Clear chocolate to avoid applying passive logic
        new_grid[choc_r][choc_c] = update_label(new_grid[choc_r][choc_c], "empty")
    elif (is_choc1 and not is_spec2) or (is_choc2 and not is_spec1):
        
        choc_r, choc_c = (r1, c1) if is_choc1 else (r2, c2)
        norm_tile = c2_label if is_choc1 else c1_label
        color = normalize_candy_name(norm_tile)
        for r in range(rows):
            for c in range(cols):
                cell = new_grid[r][c]
                if isinstance(cell, tuple):
                    _, label = cell
                else:
                    label = cell
                if color in label:
                    matched.add((r, c))
        new_grid[choc_r][choc_c] = update_label(new_grid[choc_r][choc_c], "empty")
    elif is_spec1 and is_spec2:
        s1 = get_special_suffix(c1_label)
        s2 = get_special_suffix(c2_label)
        if s1 in "VH" and s2 in "VH":
            matched.add((r1, c1))
            matched.add((r2, c2))
        elif s1 == "W" and s2 == "W":
            for center_r, center_c in [(r1, c1), (r2, c2)]:
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        rr, cc = center_r + dr, center_c + dc
                        if 0 <= rr < rows and 0 <= cc < cols:
                            matched.add((rr, cc))
            new_grid[r1][c1] = update_label(new_grid[r1][c1], "empty")
            new_grid[r2][c2] = update_label(new_grid[r2][c2], "empty")
        elif s1 == "F" and s2 == "F":
            fish_targets = set()
            for _ in range(3):
                target = spawn_fish_target(fish_targets | matched, set())
                if target:
                    fish_targets.add(target)
            matched |= fish_targets

    # 🎁 Wrapped + Striped → cross 3 rows + columns centered on the wrapped
    elif (is_spec1 and get_special_suffix(c1_label) == "W") or (is_spec2 and get_special_suffix(c2_label) == "W"):
        wrap_r, wrap_c = (r1, c1) if get_special_suffix(c1_label) == "W" else (r2, c2)

        # 3 rows centered on wrapped
        for dr in [-1, 0, 1]:
            rr = wrap_r + dr
            if 0 <= rr < rows:
                for c in range(cols):
                    matched.add((rr, c))

        # 3 cols centered on wrapped
        for dc in [-1, 0, 1]:
            cc = wrap_c + dc
            if 0 <= cc < cols:
                for r in range(rows):
                    matched.add((r, cc))

        # Clear the wrapper itself
        new_grid[wrap_r][wrap_c] = update_label(new_grid[wrap_r][wrap_c], "empty")
    
    return new_grid, matched
def is_base_candy(label):
    # Add any other special candy labels if needed
    base_colors = ["red", "blue", "green", "purple", "orange"]
    if any(color in label for color in base_colors):
        return True
    return False
def reshuffle_candies(grid):
    
    def base_label(tile):
        return tile[1] if isinstance(tile, tuple) else tile
    rows, cols = len(grid), len(grid[0])
    candy_positions = []
    candy_values = []

    # 1. Collect all normal candy tiles
    for r in range(rows):
        for c in range(cols):
            label = get_label(grid[r][c])
            if is_base_candy(label):  # custom function, defined below
                candy_positions.append((r, c))
                candy_values.append(grid[r][c])

    # 2. Shuffle candy tiles
    shuffle(candy_values)

    # 3. Reassign shuffled candies to the same positions
    for idx, (r, c) in enumerate(candy_positions):
        new_label = candy_values[idx]
        new_label = base_label(new_label)
        grid[r][c] = update_label(grid[r][c], new_label)
    return grid
def apply_move(grid, jelly_grid, r1, c1, r2, c2, tracker = None):
    """
    Performs a swap, resolves special interactions, clears matches,
    triggers updates like gravity and jelly, and returns final state.
    
    Returns:
        updated_grid, updated_jelly_grid
    """
    # Step 1: Swap with special handling
    grid, matched = apply_swap(grid, jelly_grid, r1, c1, r2, c2)

    # Step 2: If no matches, revert swap
    if matched:
        grid, jelly_grid = clear_matches(grid, jelly_grid, matched, tracker = tracker)

    # Step 4: Trigger any wrapped explosions (like explode_blue)
    grid, jelly_grid, _ = trigger_wrapped_explosions(grid, jelly_grid, tracker = tracker)

    # Step 5: Gravity/cascade/etc.
    grid, jelly_grid = update_board(grid, jelly_grid, tracker =  tracker)

    return grid, jelly_grid

def infer_hidden_jelly_layers(grid, jelly_grid, objective_targets):
    """
    Infers how many jelly layers are hidden under bubblegum by comparing
    the jelly count to the 'glass' objective.

    Returns:
        hidden_jelly_grid: 2D grid of 0, 1, or 2 indicating jelly layers under each tile
    """
    rows, cols = len(grid), len(grid[0])
    visible_jelly_count = sum(1 for row in jelly_grid for cell in row if cell)
    bubblegum_tiles = [(r, c) for r in range(rows) for c in range(cols)
                       if normalize_candy_name(grid[r][c])[:9] == 'bubblegum']

    bubblegum_count = len(bubblegum_tiles)
    glass_target = objective_targets.get("glass", 0)

    hidden_jelly_per_tile = 0
    if visible_jelly_count + bubblegum_count >= glass_target:
        hidden_jelly_per_tile = 1
    elif visible_jelly_count + 2 * bubblegum_count == glass_target:
        hidden_jelly_per_tile = 2
    else:
        # Fallback: estimate how many layers are needed per tile
        hidden_jelly_per_tile = 2

    for r, c in bubblegum_tiles:
        jelly_grid[r][c] = jelly_grid[r][c] + hidden_jelly_per_tile

    return jelly_grid