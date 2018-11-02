'''
This program is for developing the frame base classification of the lane following rcjbot
pass in a picture file, and it will be processed. The output will return a marked picture
along with the verbal output describing the movement that should be doneself.
'''
'''
FLOW GRAPH
raw input -> grayscale -> threshold -> Edge Detection -> Hough line transform -> Slope detection
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
SLOPE_EPSILON = 0.00001

# Define global constant
WIDTH = None
HEIGHT = None

# get_arguments returns the path of the target picture
def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required = True, help = 'enter the image path')
    ap.add_argument('-s', '--size', required = True, help = 'scale factor of output')
    args = vars(ap.parse_args())        # take in the arguments
    return args

# change the coordinate from upleft to down left
def toCartesian(x,y,h):
    c_y = h - y
    c_x = x
    return c_x, c_y

def main():
    args = get_arguments()
    path = str(args['image'])
    scale = int(args['size'])

    raw_img = cv2.imread(path)

    # image scaling
    HEIGHT, WIDTH = raw_img.shape[:2]       # get the geometry of the image
    height = int(HEIGHT/scale)
    width = int(WIDTH/scale)
    raw_img = cv2.resize(raw_img.copy(), (width, height))

    grey_img = cv2.cvtColor(raw_img.copy(), cv2.COLOR_BGR2GRAY)     # convert the color to grayscale
    _,threshed_img = cv2.threshold(grey_img.copy(),THRESHED_LOW,THRESHED_HIGH,cv2.THRESH_BINARY_INV)     # threshold inversely (make black white and white back)
    edged_img = cv2.Canny(threshed_img.copy(),CANNY_THRESH_1,CANNY_THRESH_2)        # edge detection
    lines_detected = cv2.HoughLinesP(edged_img.copy(),HOUGHLINE_D_RESOLUTION,HOUGHLINE_A_RESOLUTION,HOUGHLINE_MIN_LINE_LEN)     # line_segment detection

    # DRAW OUT all the lines found on the piciture
    slopes = []
    print('FOUND ' + str(len(lines_detected)) + ' LINES')
    if len(lines_detected)>0:
        for line_seg in lines_detected:
            x1, y1, x2, y2 = line_seg[0]
            cv2.line(raw_img, (x1,y1), (x2,y2), (0,255,0), 5)
            slopes.append((x2-x1)/((y2-y1)+SLOPE_EPSILON))
    np_slopes = np.array(slopes)
    np_slopes = np.arctan(np_slopes.copy())
    masked_slopes = np.ma.masked_equal(np_slopes, 0)        # masked out slopes that equals to zero
    print("Average Slope: " + str(np.mean(masked_slopes)*180/np.pi))

    cv2.imshow('output', raw_img)       # show the outputs
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
