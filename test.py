from picamera2 import Picamera2, Preview
import time

picam2 = Picamera2()

# Get full sensor resolution
camera_config = picam2.create_preview_configuration(main={"size": (2304, 1296)})  # example for Camera Module 3

picam2.configure(camera_config)
picam2.start_preview(Preview.QTGL)
picam2.start()

time.sleep(2)
picam2.capture_file("test.jpg")