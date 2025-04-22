import cv2
import numpy as np
import pyautogui
import keyboard
from PIL import Image
import time

def capture_and_process():
    screenshot = pyautogui.screenshot()
    image = np.array(screenshot)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format

    # 2. Preprocess
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)

    # 3. Edge detection
    edges = cv2.Canny(thresh, 50, 150)

    # 4. Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Filter tile-like contours
    tile_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 20 < w < 80 and 20 < h < 80:  # Adjust based on resolution
            tile_contours.append((x, y, w, h))

    # 6. Get bounding box of full board area
    if tile_contours:
        xs = [x for x, y, w, h in tile_contours]
        ys = [y for x, y, w, h in tile_contours]
        x2s = [x + w for x, y, w, h in tile_contours]
        y2s = [y + h for x, y, w, h in tile_contours]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(x2s), max(y2s)

        board_image = image[y1:y2, x1:x2]

        # Show the extracted board image
        cv2.imwrite("candy_board.png", board_image)
    else:
        print("No candy-like tiles detected. Try adjusting contour filter.")
if __name__ == "__main__":
    print("Press SPACE to capture and analyze the screen (or ESC to cancel)")
    while True:
        if keyboard.is_pressed('space'):
            capture_and_process()
            break
        elif keyboard.is_pressed('esc'):
            print("🚫 Exiting...")
            break
        time.sleep(0.1)  # Prevent high CPU usage