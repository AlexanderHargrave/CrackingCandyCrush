import pyautogui
import keyboard
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN

if __name__ == "__main__":
    print("Press SPACE to capture and analyze the screen (or ESC to cancel)")
    count = 51
    while True:
        if keyboard.is_pressed('space'):
            screenshot = pyautogui.screenshot()
            screenshot.crop((0.3*screenshot.width, 0, 0.7*screenshot.width, screenshot.height))
            screenshot.save("data//temp"+str(count)+".png")
            print("✅ Saved: data/temp"+str(count)+".png")
            count += 1
        elif keyboard.is_pressed('esc'):
            print("🚫 Exiting...")
            break