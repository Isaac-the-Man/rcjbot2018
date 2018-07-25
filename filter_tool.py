# This tool is for previewing an image or stream video with some of the basic filters
# you have to specify the source of the image if image mode is chosen
# the default mode is webcam aka video stream
# for now it supports the following filter: grey_scale thresholding,

import cv2
import argparse


def callback(val):       # this null function has no actual use
    pass

def setup_trackbars():
    cv2.namedWindow('Trackbars', cv2.WINDOW_AUTOSIZE)       # make sure that the window size is normal

    for i in ['Min', 'Max']:
        v = 0 if i == 'Min' else 255
        cv2.createTrackbar('%s' %i, 'Trackbars', v, 255, callback)      # create the tracbars

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--filter', required = True, help = 'choose the filter: threshold')
    ap.add_argument('-i', '--image', required = False, help = 'enter the image path')

    args = vars(ap.parse_args())        # take in the arguments

    return args

def get_trackbar_values():      # return the trackbar values
    values = []

    for i in ['Min', 'Max']:
        v = cv2.getTrackbarPos('%s' %i, 'Trackbars')
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

    setup_trackbars()       # initialize the trackbar

    while True:
        if video_mode:
            _, frame = vid.read()

        min, max = get_trackbar_values()        # update the value of the trackbar\

        if args['filter'] == 'threshold':
            frame2 = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
            _, output = cv2.threshold(frame2.copy(), min, max, cv2.THRESH_BINARY_INV)        # threshold the frame or image

        cv2.imshow('output', output)        # show the output image

        if cv2.waitKey(1) & 0xFF is ord('q'):       # waiting for the user to quit
            break


if __name__ == '__main__':      # run main
    main()
