import pyautogui
import keyboard
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN

def capture_and_process():
    """Capture the screen and process to detect Candy Crush game board."""
    print("📸 Capturing screen...")
    screenshot = pyautogui.screenshot()
    img = np.array(screenshot)
    
    # Convert to HSV color space for better segmentation
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Define color ranges for candies (you may need to adjust the range to match candy colors)
    lower_candy = np.array([0, 100, 100])   # Lower bound of HSV for candy colors
    upper_candy = np.array([30, 255, 255])  # Upper bound of HSV for candy colors

    # Create a mask that only includes candy colors
    candy_mask = cv2.inRange(img_hsv, lower_candy, upper_candy)

    # Perform morphological operations to clean up small noise (optional)
    kernel = np.ones((5, 5), np.uint8)
    candy_mask = cv2.morphologyEx(candy_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours of the segmented candy areas
    contours, _ = cv2.findContours(candy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours that are too small (noise) or too large (we only want tiles)
    candy_candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100 and area < 3000:  # Adjust based on candy size
            x, y, w, h = cv2.boundingRect(contour)
            candy_candidates.append((x, y, w, h))

    if len(candy_candidates) == 0:
        print("⚠️ No candy-like contours found.")
        return screenshot

    print(f"🧩 Found {len(candy_candidates)} potential candy tiles.")

    # Cluster the detected candy positions using DBSCAN
    candy_positions = np.array([(x + w // 2, y + h // 2) for (x, y, w, h) in candy_candidates])
    clustering = DBSCAN(eps=50, min_samples=5).fit(candy_positions)
    labels = clustering.labels_

    # Filter out noise points (label = -1 means noise)
    valid_positions = candy_positions[labels != -1]
    if len(valid_positions) == 0:
        print("⚠️ No valid clustered positions.")
        return screenshot

    # Get the bounding box of the clustered positions
    x_min, y_min = np.min(valid_positions, axis=0)
    x_max, y_max = np.max(valid_positions, axis=0)

    # Draw bounding box on the original image
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    # Crop the detected game board region
    cropped_board = screenshot.crop((x_min, y_min, x_max, y_max))
    cropped_board.save("game_board_cropped.png")
    print("✅ Saved: game_board_cropped.png")

    # Debug: Save the image with the bounding box
    debug_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(debug_img).save("debug_board.png")
    print("🖼 Saved: debug_board.png")

    return cropped_board

# --- Run the process when spacebar is pressed ---
if __name__ == "__main__":
    print("Press SPACE to capture and analyze the screen (or ESC to cancel)")
    while True:
        if keyboard.is_pressed('space'):
            capture_and_process()
            break
        elif keyboard.is_pressed('esc'):
            print("🚫 Exiting...")
            break
