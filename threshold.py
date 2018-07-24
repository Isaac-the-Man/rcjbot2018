import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
from time import sleep


def callback(val):
    pass

def setup_trackbars():
    cv2.namedWindow('Trackbars', 0)

    for i in ['Min', 'Max']:
        if i == 'Min':
            value = 0
        else:
            value = 255
        cv2.createTrackbar('%s' %i, "Trackbars", value, 255, callback)

def get_trackbar_values():
    values = []

    for i in ['Min', 'Max']:
        v = cv2.getTrackbarPos('%s' %(i), "Trackbars")
        values.append(v)

    return values


print('Watching You Right Now')

my_camera = PiCamera()      # initializing the camera module
my_camera.resolution = (640,480)        # resolution of the camera
my_camera.framerate = 32        # framerate of the camera
rawCapture = PiRGBArray(my_camera, size = my_camera.resolution)     # setting up the interfacing option
setup_trackbars()
sleep(1)       # wait one second for everything to settle down

for frame in my_camera.capture_continuous(rawCapture, format = 'bgr', use_video_port = True):
    raw_img = frame.array.copy()      # getting the image from the camera
    grey_img = cv2.cvtColor(raw_img.copy(), cv2.COLOR_BGR2GRAY)
    input = get_trackbar_values()
    _, thresh_img = cv2.threshold(grey_img.copy(), input[0], input[1], cv2.THRESH_BINARY_INV)
    #cv2.imshow("Original", raw_img)
    cv2.imshow("Threshold", thresh_img)

    key = cv2.waitKey(1)&0xFF       # setting 'q' as the signal for exiting
    rawCapture.truncate(0)
    if key == ord('q'):     # if the key is 'q'
        break       # get out of the capturing loop

cv2.destroyAllWindows()     # make sure that everything is closed and clean
