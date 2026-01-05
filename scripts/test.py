from picamera2 import Picamera2
import time

picam2 = Picamera2()
config = picam2.create_still_configuration()
picam2.configure(config)

picam2.start()
time.sleep(2)

picam2.capture_file("ok.jpg")
print("OK")

picam2.stop()