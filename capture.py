#!/usr/bin/env python3
from picamera2 import Picamera2
from datetime import datetime
import time
import os

# Folder unde salvăm imaginile
SAVE_DIR = "/home/mircea/Desktop/IntruderNet/images"

# Creăm folderul dacă nu există
os.makedirs(SAVE_DIR, exist_ok=True)

# Inițializăm camera
picam2 = Picamera2()
config = picam2.create_still_configuration()
picam2.configure(config)
picam2.start()

print("Camera pornită, captură 1 imagine/secunda...")

try:
    while True:
        # Timestamp pentru numele fișierului
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_DIR, f"capture_{timestamp}.jpg")
        
        # Capturează imagine
        picam2.capture_file(filename)
        print(f"Salvat: {filename}")
        
        # Așteaptă 1 secundă
        time.sleep(1)

except KeyboardInterrupt:
    print("Oprire script...")
finally:
    picam2.stop()