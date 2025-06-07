import pyautogui
import keyboard
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from random import randint
if __name__ == "__main__":
    print("Press SPACE to capture and analyze the screen (or ESC to cancel)")
    count = 127
    random = randint(1,5)
    while True:
        if keyboard.is_pressed('space'):
            if count % 5 == 1:
                random = randint(0,4)
            print(random, count - count % 5)
            screenshot = pyautogui.screenshot()
            screenshot = screenshot.crop((0.3*screenshot.width, 0, 0.7*screenshot.width, screenshot.height))
            if count - count % 5 + random == count:
                screenshot.save("data//images//val//board"+str(count)+".png")
                print("✅ Saved: data/images/val/board"+str(count)+".png")
            else:
                screenshot.save("data//images//train//board"+str(count)+".png")
                print("✅ Saved: data/images/train/board"+str(count)+".png")
               
            count += 1
        elif keyboard.is_pressed('esc'):
            print("Exiting...")
            break



