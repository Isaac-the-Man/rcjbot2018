# image walker is a testing tool for developers
# input a image to immitate the frame of the robot
# let the code here analyze and walk the robot using dots

import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt


weight_shift = float(1/250)
weight_tan = float(1/3)

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required = True, help = 'enter the image path')

    args = vars(ap.parse_args())        # take in the arguments

    return args

def cut_img(image):     # cut the image
    height, width = image.shape[:2]       # get the geometry of the image
    slice_h = int(height/2)
    for i in range(2):      # cut the image into up and down part
        up_img = image[0:slice_h, 0:width]
        down_img = image[slice_h:height, 0:width]

    return up_img, down_img

def get_midpoint(lines):        # get the midpoint of every slope produced by hough transform
    mid_points = []

    if len(lines)>0:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            mid_X = int((x1 + x2)/2)
            mid_y = int((y1 + y2)/2)
            mid_points.append((mid_X, mid_y))       # add the coordinates of the midpoint as a  tuple list

    return mid_points

def get_angle(lines):       # get the angle of the car using tangent
    slopes = []

    if len(lines)>0:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = float((x2 - x1)/(y2 - y1))        # calculate the individual slope
            slopes.append(slope)
        np_slopes = np.array(slopes)
        np_arctan = np.arctan(np_slopes)        # the the numpy array of angles
        np_arctan = np.round(np_arctan, 5)      # round to the hundreth place

    return np_arctan.tolist()       # return the angles as a python lsit

def show_vote(mid_points, angles):        # show the voting graph for maneuvering
    shift = []
    arctan = [float(angle*weight_tan) for angle in angles]

    for points in mid_points:
        x = points[0]
        y = points[1]
        x -= 320        # this sets the origin of the coordinate to the car itself
        x = x*(-y)*weight_shift
        shift.append(x)     # only x coordinate will affect the decision of shifting

    if len(angles) == len(shift):       # this should be equal since they came from the same peice of information
        plt.scatter(shift, arctan)      # graph them on the vote graph in the form of a scatter plot
        print(arctan)
        plt.xticks([-300,0,300],['L','MID','R'])
        plt.yticks([-1.7,0,1.7],['-tan(-90)','straight','tan(90)'])
        plt.grid()

def main():

    args = get_arguments()      # get the path
    path = str(args['image'])

    raw_img = cv2.imread(path)      # read the image
    resized_img = cv2.resize(raw_img.copy(), (640,480))      # resize the image to get a stable input

    up_img, resized_img = cut_img(resized_img.copy())       # resized_img here is the down_img

    #hsv_img = cv2.cvtColor(resized_img.copy(), cv2.COLOR_BGR2HSV)     # get the hsv channel
    grey_img = cv2.cvtColor(resized_img.copy(), cv2.COLOR_BGR2GRAY)       # get the grey channel

    # cv2.threshold(file, min, max, cv2.THRESH_BINARY_INV)
    _, threshed_img = cv2.threshold(grey_img.copy(), 85, 255, cv2.THRESH_BINARY_INV)        # get the black line before edge detection
    threshed_img = cv2.erode(threshed_img.copy(), None, iterations = 2)
    threshed_img = cv2.dilate(threshed_img.copy(), None, iterations = 2)

    # cv2.Canny(file, kernelwidth, kernelheight)
    edged_img = cv2.Canny(threshed_img.copy(), 100, 100)     # find the edges using canny

    # cv2.HoughLinesP(file, distance resolution, angle resolution, threshold, minLineLength, maxLineGap)
    lines = cv2.HoughLinesP(edged_img, 1, np.pi/180, 20, minLineLength = 3)     # get the lines

    # render out all the linse on the raw_img
    line_list = []
    if len(lines)>0:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(resized_img, (x1,y1), (x2,y2), (0,255,0), 5)

    mid_points = get_midpoint(lines)        # get the midpoints
    if len(mid_points) > 0:
        for points in mid_points:
            cv2.circle(resized_img, points, 1, (255,0,0))      # draw the midpoints

    angles = get_angle(lines)        # get the angle state of the car

    show_vote(mid_points, angles)

    # cocatenate the up_img and down_img back
    resized_img = np.concatenate((up_img, resized_img.copy()), axis = 0)

    cv2.imshow('output', resized_img)       # show the outputs
    cv2.imshow('canny', edged_img)
    cv2.imshow('thresholded', threshed_img)
    plt.show()      # show the vote graph

    cv2.waitKey(0)      # wait for it to end

    cv2.destroyAllWindows()

if __name__ == '__main__':      # run main
    main()
