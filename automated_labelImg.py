import pyautogui
from pynput import keyboard
from pynput.keyboard import Key, Listener
import time

# Arrow key direction map
direction_map = {
    Key.up: (0, -65),
    Key.down: (0, 65),
    Key.left: (-60, 0),
    Key.right: (60, 0)
}

def perform_action(dx, dy):
    # Hold right click
    pyautogui.mouseDown(button='right')

    # Move in direction
    pyautogui.moveRel(dx, dy, duration=0.2)

    # Release right click
    pyautogui.mouseUp(button='right')

    time.sleep(0.05)

    # Move down-right
    pyautogui.moveRel(5, 5, duration=0.05)
    pyautogui.click(button='left')

    # Move up-left
    pyautogui.moveRel(-5, -5, duration=0.05)
    pyautogui.click(button='left')

def on_press(key):
    if key == Key.esc:
        return False  # Exit listener

    if key in direction_map:
        dx, dy = direction_map[key]
        perform_action(dx, dy)

def main():
    print("Use arrow keys to trigger action. Press ESC to exit.")
    with Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == "__main__":
    main()
