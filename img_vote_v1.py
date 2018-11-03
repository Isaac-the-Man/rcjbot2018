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
SHIFT_K = 0.7
ANGLE_K = 1
ACTIVATION_BIAS = 0
DIRECTION_DIAL_LEN = 100

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

def activation_func(x):
    y = (np.pi/2)/(1+(np.e)**(-14*(x-0.5)))
    return (y + ACTIVATION_BIAS)

def get_direction_vector(img, angle, shift, h, w):
    shift_scale = 0
    angle_scale = 0
    final_turn = 0
    cv2.line(img, (int(w/2), 0), (int(w/2), h), (0,0,255), 1)       # this is the middle line

    # scale with constant
    if (angle*shift > 0):       # if there is a shift vector
        shift_scale = SHIFT_K*(shift/(w/2))
        print('SHIFT INCLUDED')
    angle_scale = ANGLE_K*(angle/90)

    # add and scale back to one
    added_scale = (shift_scale + angle_scale)/1
    print('Angle Scale: ' + str(angle_scale))
    print('Shift Scale: ' + str(shift_scale))
    print('Added Scale: ' + str(added_scale))

    #activate and times the direction back
    if (added_scale < 0):
        final_turn = activation_func(abs(added_scale))*-1
    else:
        final_turn = activation_func(abs(added_scale))

    # to cartesian and origin shift
    x_final = int(w/2 + DIRECTION_DIAL_LEN*np.cos(np.pi/2-final_turn))
    y_final = int(h - DIRECTION_DIAL_LEN*np.sin(np.pi/2-final_turn))

    cv2.line(img, (int(w/2), h), (x_final,y_final), (0,0,255), 5)

    print(final_turn*180/np.pi)
    print((x_final,  y_final))

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

    # DRAW OUT all the lines found on the piciture and find the slope and find the average shift
    slopes = []
    midpoints = []
    print('FOUND ' + str(len(lines_detected)) + ' LINES')
    if len(lines_detected)>0:
        for line_seg in lines_detected:
            x1, y1, x2, y2 = line_seg[0]
            cv2.line(raw_img, (x1,y1), (x2,y2), (0,255,0), 5)
            slopes.append((x2-x1)/((y2-y1)+SLOPE_EPSILON))
            midpoints.append((x1+x2)/2 - WIDTH/2)
    np_slopes = np.array(slopes)
    np_midpoints = np.array(midpoints)
    np_slopes = np.arctan(np_slopes.copy())
    masked_slopes = np.ma.masked_equal(np_slopes, 0)        # masked out slopes that equals to zero

    avg_angle = np.mean(masked_slopes)*-180/np.pi
    avg_shift = np.mean(np_midpoints)
    print("Average Slope: " + str(avg_angle) + " Degree")        # average out the slope
    print('Average Shift: ' + str(avg_shift))
    '''
    Draw a vertical bisector. The average angle is the angle between the bisector and the desired direction.
    negative angle is to the right and positive is to the left. For shift, negative is to the left and positive to the right
    '''

    get_direction_vector(raw_img, avg_angle, avg_shift, height, width)
    cv2.imshow('output', raw_img)       # show the outputs
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
