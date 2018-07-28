# This tool is for previewing an image or stream video with some of the basic filters
# you have to specify the source of the image if image mode is chosen
# the default mode is webcam aka video stream
# for now it supports the following filter: grey_scale thresholding, and HSV
# example format for video: python filter_tool.py -f hsv
# example format for image: python filter_tool.py -f hsv -i D:/home/img.jpg

import cv2
import argparse


def callback(val):       # this null function has no actual use
    pass

def setup_trackbars(filter):        # take in filter to decide the format of the trackbar
    cv2.namedWindow('Trackbars', cv2.WINDOW_AUTOSIZE)       # make sure that the window size is normal

    for i in ['Min', 'Max']:
        v = 0 if i == 'Min' else 255
        if filter == 'THRESHOLD':
            cv2.createTrackbar('%s' %i, 'Trackbars', v, 255, callback)      # create the tracbars for thresholding
        elif filter == 'HSV':
            for j in str(filter):
                cv2.createTrackbar('%s_%s' %(i, j), 'Trackbars', 180 if i == 'Max' and j == 'H' else v, 255 if j != 'H' else 180, callback)      # trackbar for hsv

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--filter', required = True, help = 'choose the filter: threshold')
    ap.add_argument('-i', '--image', required = False, help = 'enter the image path')

    args = vars(ap.parse_args())        # take in the arguments

    return args

def get_trackbar_values(filter):      # return the trackbar values
    values = []

    for i in ['Min', 'Max']:
        if filter == 'THRESHOLD':
            v = cv2.getTrackbarPos('%s' %i, 'Trackbars')
            values.append(v)
        elif filter == 'HSV':
            for j in str(filter):
                v = cv2.getTrackbarPos('%s_%s' %(i, j), 'Trackbars')
                values.append(v)

    return values

def main():
    video_mode = True
    frame = None
    output = None

    args = get_arguments()

    if args['image'] != None:       # read the image if path is given
        frame = cv2.imread(str(args['image']))
        video_mode = False
    else:
        vid = cv2.VideoCapture(0)     # if no img path is given, stream vido mode will be on

    setup_trackbars(args['filter'].upper())       # initialize the trackbar

    while True:
        if video_mode:      # read the frame of the video stream
            _, frame = vid.read()

        value = get_trackbar_values(args['filter'].upper())        # update the value of the trackbar

        if args['filter'].upper() == 'THRESHOLD':       # threhsolding selected
            frame2 = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
            _, output = cv2.threshold(frame2.copy(), value[0], value[1], cv2.THRESH_BINARY_INV)        # threshold the frame or image
        elif args['filter'].upper() == 'HSV':       # hsv filter chosen
            frame2 = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)      # cvt the color to hsv
            output = cv2.inRange(frame2, (value[0], value[1], value[2]), (value[3], value[4], value[5]))        # mask out the hsv color range

        cv2.namedWindow('output', cv2.WINDOW_AUTOSIZE)      # create the window
        cv2.imshow('output', output)        # show the output image

        if cv2.waitKey(1) & 0xFF is ord('q'):       # waiting for the user to quit
            break


if __name__ == '__main__':      # run main
    main()
