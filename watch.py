# watch is an application for previewing the PiCamera
# It has no further functions other than that
# press 'q' to quit the application

import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
from time import sleep


print('Watching You Right Now')

my_camera = PiCamera()      # initializing the camera module
my_camera.resolution = (640,480)        # resolution of the camera
my_camera.framerate = 32        # framerate of the camera
rawCapture = PiRGBArray(my_camera, size = my_camera.resolution)     # setting up the interfacing option
sleep(1)       # wait one second for everything to settle down

for frame in my_camera.capture_continuous(rawCapture, format = 'bgr', use_video_port = True):
    raw_image = frame.array.copy()      # getting the image from the camera

    cv2.imshow('Stream', raw_image)     # showing of the image
    key = cv2.waitKey(1)&0xFF       # setting 'q' as the signal for exiting
    rawCapture.truncate(0)
    if key == ord('q'):     # if the key is 'q'
        break       # get out of the capturing loop

cv2.destroyAllWindows()     # make sure that everything is closed and clean
