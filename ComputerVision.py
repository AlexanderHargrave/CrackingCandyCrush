import pyautogui
import keyboard
from PIL import Image
import numpy as np
import cv2
import pytesseract
import time
import os
from collections import defaultdict
# Configure pytesseract path (update this to your Tesseract installation path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Game board configuration (adjust based on your screen resolution)
BOARD_SIZE = 9  # 9x9 grid
CANDY_TYPES = {
    0: 'blue',
    1: 'green',
    2: 'orange',
    3: 'purple',
    4: 'red',
    5: 'yellow',
    6: 'chocolate'
}

def remove_windows_taskbar(image):
    """Remove the standard Windows taskbar from the bottom"""
    img_array = np.array(image)
    height = img_array.shape[0]
    
    # Scan bottom 100px for taskbar
    for y in range(height-1, max(height-100, 0), -1):
        row = img_array[y]
        if np.allclose(row, row[0], atol=10):  # Uniform color detection
            return image.crop((0, 0, image.width, y))
    
    # Default removal if not detected
    return image.crop((0, 0, image.width, image.height - 40))

def remove_top_white_bar(image):
    """Remove white bar from the top"""
    img_array = np.array(image)
    
    # Define white color range
    lower_white = np.array([200, 200, 200])
    upper_white = np.array([255, 255, 255])
    
    white_mask = np.all((img_array >= lower_white) & (img_array <= upper_white), axis=2)
    
    # Find first non-white row from top
    for y in range(img_array.shape[0]):
        if not np.all(white_mask[y]):
            return image.crop((0, y, image.width, image.height))
    
    return image
def find_green_boundary(image):
    """Find the left and right green boundaries separating game area from background"""
    img_array = np.array(image)
    
    # More precise green color range for Candy Crush boundaries
    # These values might need slight adjustment based on actual game colors
    lower_green = np.array([0, 100, 0])
    upper_green = np.array([100, 255, 100])
    # Create a mask for pixels within the green range
    green_mask = np.all((img_array >= lower_green) & (img_array <= upper_green), axis=2)
    
    # Find columns that contain green boundary pixels
    green_columns = np.any(green_mask, axis=0)
    
    # To handle potential noise, we'll look for continuous green sections
    # Find left boundary (first continuous green section from left)
    left_bound = np.argmax(green_columns)
    
    lower_green = np.array([50, 120, 50])   # Minimum green values (R, G, B)
    upper_green = np.array([120, 200, 120]) # Maximum green values
    # Create a mask for pixels within the green range
    green_mask = np.all((img_array >= lower_green) & (img_array <= upper_green), axis=2)
    green_columns = np.any(green_mask, axis=0)
    # Find right boundary (first continuous green section from right)
    right_bound = len(green_columns) - 1
    for i in range(len(green_columns)-1, 2, -1):
        if all(green_columns[i-2:i]):
            right_bound = i
            break
    
    # Alternative approach if the above doesn't work well:
    # left_bound = np.argmax(green_columns)
    # right_bound = len(green_columns) - np.argmax(green_columns[::-1]) - 1
    
    return left_bound+5, right_bound-5

def load_templates(template_dir):
    templates = {}
    for filename in os.listdir(template_dir):
        if filename.endswith('.webp'):
            name = os.path.splitext(filename)[0]
            path = os.path.join(template_dir, filename)
            # Load in BGR then convert to RGB
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            templates[name] = img_rgb
    return templates

def detect_game_state_shape_by_edge(image, canny_threshold1=50, canny_threshold2=150):
    """
    Crops the image to roughly the game board area, then uses Canny edge detection
    to first determine the vertical boundaries (top and bottom) where the board is found.
    
    Within this vertical band, for each horizontal candy-sized step, a horizontal line 
    (located near the center of the row) is examined for a clear vertical edge:
      - The first edge from the left is taken as left_edge.
      - The first edge from the right is taken as right_edge.
      
    The span between these defines the width of that row, from which the expected number
    of candy cells (num_cells) is computed.
    
    Returns a list of row definitions as tuples:
      (row_center_y, left_edge, right_edge, num_cells)
    and the RGB version of the cropped image.
    """
    # Crop to roughly the game board and save for debugging.
    """image = image.crop((
        image.width // 10 - 20, 
        image.height // 4 - 50, 
        int(image.width * 9 / 10) + 20, 
        int(image.height * 4 / 5) + 100
    ))"""
    image.save("debug_cropped.png")
    candy_width = 67
    candy_height = 74
    # Convert image to RGB (if not already) and get its shape.
    image_rgb = np.array(image.convert('RGB'))
    h, w, _ = image_rgb.shape

    # Convert to grayscale and blur to reduce noise.
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Compute edges with Canny.
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
    cv2.imwrite("debug_edges.png", edges)
    
    # --- Step 1: Determine the overall vertical boundaries of the game board ---
    # Sum up the edge pixels in each row.
    edge_sum = np.sum(edges > 0, axis=1)
    # Threshold to decide if a row is "edge-rich". Adjust 0.05 as needed.
    threshold_edge_count = int(w * 0.05)
    rows_with_edges = np.where(edge_sum > threshold_edge_count)[0]
    
    if len(rows_with_edges) == 0:
        print("No significant horizontal edges found!")
        top_boundary = 0
        bottom_boundary = h - 1
    else:
        top_boundary = rows_with_edges[0]
        bottom_boundary = rows_with_edges[-1]
    
    # --- Step 2: Iterate within vertical boundaries to detect each row's horizontal bounds ---
    row_bounds = []
    print(f"Top boundary: {top_boundary}, Bottom boundary: {bottom_boundary}")
    for base_y in range(top_boundary, bottom_boundary, candy_height):
        row_y = base_y + candy_height
        if row_y >= h:
            break

        # Extract the horizontal line at the current row.
        line = edges[row_y, :]  # 1D array (width,)
        left_edge, right_edge = None, None

        # From the left: find the first pixel that is an edge.
        for x in range(0, w // 2):
            if line[x] > 0:
                left_edge = x
                break

        # From the right: find the first pixel that is an edge.
        for x in range(w - 1, w // 2, -1):
            if line[x] > 0:
                right_edge = x
                break

        # Only accept the row if both edges are found and the span is significant.
        if left_edge is not None and right_edge is not None:
            span = right_edge - left_edge
            if span > candy_width * 2:  # at least two candies wide
                num_cells = int(round(span / candy_width))
                row_bounds.append((row_y, left_edge, right_edge, num_cells))
    
    print("Detected rows (y, left_edge, right_edge, num_cells):")
    print(row_bounds)
    return row_bounds, image_rgb
def match_cell_with_template_orb(cell_img, templates):
    orb = cv2.ORB_create()
    cell_gray = cv2.cvtColor(cell_img, cv2.COLOR_RGB2GRAY)
    keypoints1, descriptors1 = orb.detectAndCompute(cell_gray, None)

    best_match = None
    best_score = float('-inf')

    for name, template in templates.items():
        template_resized = cv2.resize(template, (cell_gray.shape[1], cell_gray.shape[0]))
        template_gray = cv2.cvtColor(template_resized, cv2.COLOR_RGB2GRAY)
        keypoints2, descriptors2 = orb.detectAndCompute(template_gray, None)

        if descriptors1 is None or descriptors2 is None:
            continue

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)

        if not matches:
            continue

        matches = sorted(matches, key=lambda x: x.distance)
        score = -sum([m.distance for m in matches[:10]])  # Lower distance = better match

        if score > best_score:
            best_score = score
            best_match = name

    return best_match, best_score

def detect_candies(image, template_dir='templates', debug_dir="debug_cells"):
    """
    Detect candies using ORB feature matching. Saves debug images and prints scores.
    """

    os.makedirs(debug_dir, exist_ok=True)
    candy_width = 67
    candy_height = 74

    row_bounds, image_rgb = detect_game_state_shape_by_edge(image)
    h, w, _ = image_rgb.shape
    debug_img = image_rgb.copy()

    templates = load_templates(template_dir)
    grid = []

    for row_idx, (row_y, left_edge, right_edge, num_cells) in enumerate(row_bounds):
        cv2.line(debug_img, (left_edge, row_y), (right_edge, row_y), (255, 0, 0), 2)
        boundaries = np.linspace(left_edge, right_edge, num_cells + 1, dtype=int)

        for b in boundaries:
            cv2.line(debug_img, (b, row_y), (b, row_y + candy_height), (0, 255, 0), 1)

        row_cells = []

        for i in range(num_cells):
            x1 = boundaries[i] - 5
            x2 = boundaries[i + 1] + 5
            y1 = row_y - candy_height - 5
            y2 = y1 + candy_height + 5

            if y1 < 0 or y2 > h:
                row_cells.append("out_of_bounds")
                continue

            cell_img = image_rgb[y1:y2, x1:x2]
            best_match, best_score = match_cell_with_template_orb(cell_img, templates)

            cell_filename = f"{debug_dir}/row{row_idx}_cell{i}_{best_match}_score{best_score:.2f}.png"
            Image.fromarray(cell_img).save(cell_filename)

            print(f"\nCell [row {row_idx} col {i}]: Best match: {best_match} (score {best_score:.2f})")
            row_cells.append(best_match if best_match else "empty")

        grid.append(row_cells)

    Image.fromarray(debug_img).save("debug_edge_grid.png")
    return grid

def detect_requirements(image):
    """Detect level requirements from the top of the screen"""
    # Crop the requirements area (adjust these coordinates based on your screen)
    requirements_area = image.crop((image.width//2-100, 75, image.width//2+100, 150))
    requirements_area.save("requirements_area.png")  # Save for debugging
    # Preprocess for OCR
    gray = cv2.cvtColor(np.array(requirements_area), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Use OCR to read text
    text = pytesseract.image_to_string(thresh, config='--psm 6')
    
    # Parse text to extract requirements
    requirements = {}
    if "Collect" in text:
        parts = text.split("Collect")[1].split()
        if len(parts) >= 2:
            try:
                count = int(''.join(filter(str.isdigit, parts[0])))
                color = parts[1].lower()
                requirements['collect'] = {color: count}
            except (ValueError, IndexError):
                pass
    
    return requirements

def detect_remaining_moves(image):
    """Detect remaining moves counter"""
    # Crop the moves area (usually top-right)
    moves_area = image.crop((image.width//10-20, 75, image.width//10+150, 150))
    moves_area.save("moves_area.png")  # Save for debugging
    # Preprocess for OCR
    gray = cv2.cvtColor(np.array(moves_area), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Use OCR
    moves_text = pytesseract.image_to_string(thresh, config='--psm 7 digits')
    
    try:
        return int(''.join(filter(str.isdigit, moves_text)))
    except:
        return None

def analyze_game_state(screenshot):
    """Main function to analyze the game state"""

    
    # 3. Detect candies
    candy_grid = detect_candies(screenshot)
    
    # 4. Detect requirements
    requirements = detect_requirements(screenshot)
    
    # 5. Detect moves
    moves_left = detect_remaining_moves(screenshot)
    
    return {
        'board': candy_grid,
        'requirements': requirements,
        'moves_left': moves_left,
    }

def print_game_state(state):
    """Print the game state in a readable format"""
    if not state:
        print("No game state available")
        return
    
    print("\n=== Game State ===")
    print(f"Moves Left: {state['moves_left']}")
    print("Requirements:", state['requirements'])
    
    print("\nBoard Layout:")
    for row in state['board']:
        print(" ".join([CANDY_TYPES.get(cell, '?').ljust(8) for cell in row]))

def capture_and_analyze():
    print("Press ENTER to capture and analyze the screen (or ESC to cancel)")
    
    while True:
        if keyboard.is_pressed('enter'):
            screenshot = pyautogui.screenshot()
            
            try:
                # Save original screenshot
                screenshot.save("full_screenshot.png")
                print("Saved full screenshot as full_screenshot.png")
                
                # Analyze game state
                screenshot = remove_windows_taskbar(screenshot)
                screenshot = remove_top_white_bar(screenshot)
                left_bound, right_bound = find_green_boundary(screenshot)
                screenshot = screenshot.crop((left_bound, 0, right_bound, screenshot.height))


                game_state = analyze_game_state(screenshot)
                
                if game_state:
                    print_game_state(game_state)
                    screenshot.save("cropped_game.png")
                    print("\nSaved game board as game_board.png")
                
            except Exception as e:
                print(f"Error: {e}")
                screenshot.save("full_screenshot_fallback.png")
                print("Saved full screenshot as fallback")
            
            break
            
        elif keyboard.is_pressed('esc'):
            print("Cancelled")
            break

if __name__ == "__main__":
    capture_and_analyze()