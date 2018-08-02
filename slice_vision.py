# this script is for testing only
# this script test the following lane using the method of slicing and contouring

import cv2
import numpy as np
import argparse
from time import sleep

def callback():     # do nothing
    pass

def get_arguments():        # get the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required = False, help = 'enter the image path or for image mode, or input nothing for video mode')

    args = vars(ap.parse_args())

    return args

def get_ROI(frame):      # get the region of interest
    height, width = frame.shape[:2]
    slice_h = int(height/2)     # get the half length of the frame's height
    unwanted = frame[0:slice_h,0:width]
    ROI = frame[slice_h:height,0:width]

    return unwanted, ROI        # return two cutted frame

def slice_img(ROI):     # slice the region of interest for further analysis
    slice_list = []
    slice_num = 3
    height, width = ROI.shape[:2]
    slice_h = int(height/3)      # calculate the height of each slice

    for slice in range(slice_num):
        new_slice = ROI[slice_h*slice:slice_h*slice+slice_h,0:width]    #cut the ROI into slice_num parts
        slice_list.append(new_slice)

    return slice_list       # the returned slice_list is ordered from up to down

def first_filter(frame):        # perform some basic filter such as thresholding and blurring
    raw_img = frame.copy()
    grey_img = cv2.cvtColor(raw_img.copy(), cv2.COLOR_BGR2GRAY)     # make it greyscale
    _, threshed_img = cv2.threshold(grey_img.copy(), 80, 255, cv2.THRESH_BINARY_INV)        # threshold the frame of image
    threshed_img = cv2.erode(threshed_img.copy(), None, iterations = 2)     # perform some basic noise remover
    threshed_img = cv2.dilate(threshed_img.copy(), None, iterations = 2)

    return  raw_img, threshed_img

def analyze(f_slice, uf_slice):     # analyze the slice for to get a moment point
    # cv2.Canny(file, kernelwidth, kernelheight)
    #canny_slice = cv2.Canny(f_slice.copy(), 70, 70)
    _,cnts,_ = cv2.findContours(f_slice,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)      # contour it
    main_cnts = max(cnts, key=cv2.contourArea)      # find the contour with the biggest area
    cv2.drawContours(uf_slice,cnts,-1,(0,255,0),5)

    M = cv2.moments(main_cnts)
    if M['m00']==0:
        contourCenterX = 0
        contourCenterY = 0
    else:
        contourCenterX = int(M['m10']/M['m00'])
        contourCenterY = int(M['m01']/M['m00'])

    cv2.circle(uf_slice,(contourCenterX,contourCenterY),7,(255,0,0),-1)

    return uf_slice        # return the analyzed sice

def repack_slice(unwanted, raw_slice = []):       # repack the previous sliced images back into the original image
    raw_slice.insert(0, unwanted)       # insert the unwanted image as the first element of the slice list
    slice_tuple = tuple(raw_slice)      # convert the list into tuple
    output = np.concatenate(slice_tuple, axis = 0)       # repack the image

    return output       # output is packed image of the slices

def main():
    video_mode = True   # this is the default mode, pass in argument i to change to image mode
    args = get_arguments()      # receiving the input
    if args['image'] != None:
        video_mode = False      # if image path was given, change to image mode
    else:
        cap = cv2.VideoCapture(0)      # setting live stream camera`
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)       # decide the size of the live stream
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)        # framerate

    while True: # start the capturing loop
        frame = None

        if video_mode:
            _, frame = cap.read()       # get the frame from the video stream
        else:
            frame = cv2.imread(str(args['image']))      # get the frame from the image path
            frame = cv2.resize(frame.copy(), (640,480))

        up_img, ROI_img = get_ROI(frame.copy())      # get only the region of interest to do further analyses

        ROI_img, filtered_ROI = first_filter(ROI_img.copy())      # do some basic filtering

        slice_list = slice_img(ROI_img.copy())      # slice the original image to show the result on it
        filtered_slice_list = slice_img(filtered_ROI.copy())     # slice the image into several slices for further analysis

        analyzed_list = []
        for i in range(len(slice_list)):
            analyzed_list.append(analyze(filtered_slice_list[i], slice_list[i]))        # analyze and return the slice

        repacked_img = repack_slice(up_img, analyzed_list)     # repack the image with contours

        cv2.imshow('frame', repacked_img)
        # cv2.imshow('output', slice_list[0])        # show the output of the image
        # cv2.imshow('output2', filtered_slice_list[0])        # show the output of the image

        if cv2.waitKey(1) & 0xFF is ord('q'):       # waiting for the user to quit
            break

    if video_mode:
        cap.release()       # release the resources
    cv2.destroyAllWindows()

if __name__ == '__main__':      # run main
    main()
