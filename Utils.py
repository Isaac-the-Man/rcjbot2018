# Utils provide useful functions for every other class to work,
# including interfacint with each other, acquiring information,
# and even basic computer vision functions such as filters

import cv2
from time import sleep
from infos import Filter_type


class Filter(object):
    """docstring for Filter."""
    def __init__(self):
        self.filter = None      # THis saves the filter chosed to perform
        self.Gaussian_kernels = (17, 17)        # Gausian kernels used for blurring, check the url for more info
        self.Threshold_max = 255
        self.Threshold_min = 110

    def perform(self, img):
        if self.filter == None:
            return 0
        elif self.filter == Filter_type.Grey:        # Grey filter
            new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return new_img
        elif self.filter == Filter_type.Gaussian:
            new_img = cv2.GaussianBlur(img, self.Gaussian_kernels, 0)       # Check this:https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
            return new_img
        elif self.filter == Filter_type.HSV:        # HSV filter for fidning color related objects
            new_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            return new_img
        elif self.filter == Filter_type.Bilateral:      # Bilateral filter, waiting for editing
            print('')
            return img
        elif self.filter == Filter_type.Threshold:     # wating for further editing
            new_img = cv2.threshold(img, self.Threshold_min, self.Threshold_max, cv2.THRESH_BINARY_INV)
            return new_img
        else:
            self.filter_type = None

    def set_filter_type(self, filter_type):     # Must call this function to set filter type before perform
        self.filter = filter_type
        #print('filter type set')
