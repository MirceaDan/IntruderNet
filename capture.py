#!/usr/bin/env python3
from picamera2 import Picamera2
from datetime import datetime
import time
import os
import cv2
import numpy as np

time.sleep(5)

def is_daylight(frame_path):
    gray = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    median = np.median(gray)
    return median > 20, median

def get_today_dir():
    today = datetime.now().strftime("%d-%m-%Y")
    return today

# Folder unde salvăm imaginile
SAVE_DIR = "/home/mircea/Desktop/IntruderNet/images"

# Creăm folderul dacă nu există
os.makedirs(SAVE_DIR, exist_ok=True)

# Inițializăm camera
picam2 = Picamera2()
config = picam2.create_still_configuration() #default configuration
picam2.configure(config)
picam2.start()

controls = {
    "ExposureTime": 200000,   # microsecunde (200ms)
    "AnalogueGain": 8.0,
    "NoiseReductionMode": 2,
    "AwbEnable": False
}
picam2.set_controls(controls)



print("Camera pornită, captură 1 imagine/secunda...")

try:
    while True:
        # Timestamp pentru numele fișierului
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        NEW_SAVE_DIR = SAVE_DIR + "/" + get_today_dir()
        os.makedirs(NEW_SAVE_DIR, exist_ok=True)
        filename = os.path.join(NEW_SAVE_DIR, f"capture_{timestamp}.jpg")
        
        # Capturează imagine
        picam2.capture_file(filename)
        print(f"Salvat: {filename}")
        
        # Așteaptă 1 secundă
        time.sleep(1)

        [daylight, avg] = is_daylight(filename)
        print(f"Daylight: {daylight} avg pixel density {avg}")

        if not daylight:
            os.remove(filename)
            print("Low light detected, sleeping for 10 secs...")
            time.sleep(10)

except KeyboardInterrupt:
    print("Oprire script...")
finally:
    picam2.stop()