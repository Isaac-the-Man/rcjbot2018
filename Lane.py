from time import sleep
import cv2
from Utils import Filter
from infos import Filter_type
from picamera.array import PiRGBArray
from picamera import PiCamera


print('Watching You Right Now')

my_camera = PiCamera()      # initializing the camera module
my_camera.resolution = (640,480)        # resolution of the camera
my_camera.framerate = 32        # framerate of the camera
rawCapture = PiRGBArray(my_camera, size = my_camera.resolution)     # setting up the interfacing option
my_filter = Filter()
sleep(1)       # wait one second for everything to settle down

for frame in my_camera.capture_continuous(rawCapture, format = 'bgr', use_video_port = True):
    raw_img = frame.array.copy()      # getting the image from the camera

    # field filters
    my_filter.set_filter_type(Filter_type.Gaussian)
    blur_img = my_filter.perform(raw_img.copy())
    my_filter.set_filter_type(Filter_type.Grey)
    grey_img = my_filter.perform(blur_img.copy())
    my_filter.set_filter_type(Filter_type.Threshold)
    _, thresh_img = my_filter.perform(grey_img.copy())

    cv2.imshow('Stream', thresh_img)     # showing of the image
    key = cv2.waitKey(1)&0xFF       # setting 'q' as the signal for exiting
    rawCapture.truncate(0)
    if key == ord('q'):     # if the key is 'q'
        break       # get out of the capturing loop

cv2.destroyAllWindows()     # make sure that everything is closed and clean
