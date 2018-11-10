'''
This program is the adaptation of img_vote_v2.py to live processing. Input a camera input and the programe will
process it along with marks that shows the direction that should be followed by the rcjbot.
'''

import cv2
import numpy as np
import argparse
from img_vote_v2 import process

# Define hyperparameters here
PROCESS_RATE = 1 # once per NUM frame


def main():
    # setup camera
    vid = cv2.VideoCapture(0)       # set the video mode
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)       # decide the size of the live stream
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    vid.set(cv2.CAP_PROP_FPS, 30)        # framerate

    counter = 0
    while True:
        counter += 1
        _, output = vid.read()      # read out the frame

        if counter == PROCESS_RATE:
            #output = process(frame, 1)
            output = process(output.copy(), 1)
            counter = 0

        cv2.imshow('output', output)        # show the output image

        if cv2.waitKey(1) & 0xFF is ord('q'):       # waiting for the user to quit
            break
    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
