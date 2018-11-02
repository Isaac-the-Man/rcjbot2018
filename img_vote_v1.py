'''
This program is for developing the frame base classification of the lane following rcjbot
pass in a picture file, and it will be processed. The output will return a marked picture
along with the verbal output describing the movement that should be doneself.
'''

# import essentail pacakages
import cv2
import numpy as np
from matplotlib import pyplot
import argparse

# Define any hyperparameters here
THRESHED_LOW = 85
THRESHED_HIGH = 255
CANNY_THRESH_1 = 100
CANNY_THRESH_2 = 100
HOUGHLINE_D_RESOLUTION = 1
HOUGHLINE_A_RESOLUTION = np.pi/180
HOUGHLINE_THRESH = 20
HOUGHLINE_MIN_LINE_LEN = 20

# get_arguments returns the path of the target picture
def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required = True, help = 'enter the image path')
    args = vars(ap.parse_args())        # take in the arguments
    return args

def main():
    args = get_arguments()
    path = str(args['image'])
    raw_img = cv2.imread(path)
    # can add resize image here
    grey_img = cv2.cvtColor(raw_img.copy(), cv2.COLOR_BGR2GRAY)     # convert the color to grayscale
    thresehed_img = cv2.threshold(grey_img.copy(),THRESHED_LOW,THRESHED_HIGH,cv2.THRESH_BINARY_INV)     # threshold inversely (make black white and white back)
    edged_img = cv2.Canny(threshed_img.copy(),CANNY_THRESH_1,CANNY_THRESH_2)        # edge detection
    lines_detected = cv2.HoughLinesP(edged_img.copy(),HOUGHLINE_D_RESOLUTION,HOUGHLINE_A_RESOLUTION,HOUGHLINE_MIN_LINE_LEN)     # line_segment detection

    # DRAW OUT all the lines found on the piciture
    print('FOUND ' + str(len(lines_detected)) + ' LINES')
    if len(lines_detected)>0:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(raw_img, (x1,y1), (x2,y2), (0,255,0), 5)


if __name__ == '__main__':
    main()
