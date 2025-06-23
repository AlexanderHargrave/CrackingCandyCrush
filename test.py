import pyautogui
import keyboard
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from random import randint
if __name__ == "__main__":
    print("Press SPACE to capture and analyze the screen (or ESC to cancel)")
    count = 29
    while True:
        if keyboard.is_pressed('space'):

            screenshot = pyautogui.screenshot()
            screenshot = screenshot.crop((0.3*screenshot.width, 0, 0.7*screenshot.width, screenshot.height))
            screenshot.save("data//test//images//test"+str(count)+".png")
            print("✅ Saved: data/temp/images/test"+str(count)+".png")    
            # create an empty json file
            with open("data//test//labels//test"+str(count)+".txt", "w") as f:
                f.write("")        
            count += 1
        elif keyboard.is_pressed('esc'):
            print("Exiting...")
            break



