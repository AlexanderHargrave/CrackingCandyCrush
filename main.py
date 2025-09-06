import pyautogui
import keyboard
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from random import randint
from candy_vision_train import predict_optimal_move
if __name__ == "__main__":
    print("Press SPACE to capture and analyze the screen (or ESC to cancel)")
    start = True
    while start:
        if keyboard.is_pressed('space'):
            screenshot = pyautogui.screenshot()
            screenshot = screenshot.crop((0.3*screenshot.width, 0, 0.7*screenshot.width, screenshot.height))
            screenshot_path = "current_image.png"
            screenshot.save(screenshot_path)
            move, tracker, location1, location2 = predict_optimal_move(screenshot_path)
            centreX1 = (location1[0] + location1[2])//2
            centreY1 = (location1[1] + location1[3]) // 2
            centreX2 = (location2[0] + location2[2])//2
            centreY2 = (location2[1] + location2[3]) // 2
            screen_width, screen_height = pyautogui.size()
            x_shift = 0.3 * screen_width
            centreX1 += x_shift
            centreX2 += x_shift
            pyautogui.moveTo(centreX1, centreY1, duration=0.2)
            pyautogui.dragTo(centreX2, centreY2, duration=0.2, button='left')
            print("Move: ", move)
            print("Tracker1: ", tracker)
     
        elif keyboard.is_pressed('esc'):
            print("Exiting...")
            start = False



