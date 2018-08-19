import cv2
import numpy as np
from matplotlib import pyplot as plt


weight_shift = float(1/250)     # for altering the shift received
weight_tan = float(1/180)       # for altering the tan received
tan_digit = int(5)

def get_ROI(image):     # get the region of interest, in this case, ROI is the bottom half of the image
    height, width = image.shape[:2]       # get the geometry of the image
    slice_h = int(height/2)
    for i in range(2):      # cut the image into up and down part
        up_img = image[0:slice_h, 0:width]
        ROI_img = image[slice_h:height, 0:width]

    return up_img, ROI_img

def get_midpoint(lines):        # get the midpoint of every slope produced by hough transform
    mid_points = []

    if len(lines)>0:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            mid_X = int((x1 + x2)/2)
            mid_y = int((y1 + y2)/2)
            mid_points.append((mid_X, mid_y))       # add the coordinates of the midpoint as a  tuple list

    return mid_points

def get_angles(lines):       # get the angle of the car using tangent
    slopes = []
    avg_ys = []

    if len(lines)>0:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = float((x2 - x1)/(y2 - y1))        # calculate the individual slope
            avg_y = float((y2 + y1)/2)
            slopes.append(slope)
            avg_ys.append(avg_y)
        np_slopes = np.array(slopes)
        np_arctan = np.arctan(np_slopes)        # the the numpy array of angles
        np_arctan = np.round(np_arctan, tan_digit)      # round to the hundreth place

    return np_arctan.tolist(), avg_ys       # return the angles as a python lsit

def get_weighted(mid_points, angles, angles_ys):        # calculate the weighted results
    assert len(mid_points) == len(angles)        # this should be equal
    weighted_shifts =[]
    weighted_tans = []

    for mid_point in mid_points:
        x = mid_point[0]
        y = mid_point[1]
        x -= 320        # translate to machine state shift
        weighted_shift = -x*y*weight_shift
        weighted_shifts.append(weighted_shift)

    for i in range(len(angles)):
        weighted_tan = float(angles[i]*angles_ys[i]*weight_tan)
        #print('%s x %s x %s' %(str(angles[i]),str(angles_ys[i]),str(weight_tan)))
        weighted_tans.append(weighted_tan)

    return weighted_shifts, weighted_tans

def show_vote(w_shifts, w_tans):
    plt.scatter(w_shifts, w_tans)      # graph them on the vote graph in the form of a scatter plot
    plt.xticks([-300,0,300],['L','MID','R'])
    plt.yticks([-1.7,0,1.7],['-tan(-90)','straight','tan(90)'])
    plt.grid()
    plt.ion()

def main():

    vid = cv2.VideoCapture(0)       # set the video mode
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)       # decide the size of the live stream
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    vid.set(cv2.CAP_PROP_FPS, 30)        # framerate

    while True:
        _, frame = vid.read()      # read out the frame
        up_img, ROI_img = get_ROI(frame.copy())        # cut the image into two part
        grey_img = cv2.cvtColor(ROI_img.copy(), cv2.COLOR_BGR2GRAY)       # get the grey channel

        # cv2.threshold(file, min, max, cv2.THRESH_BINARY_INV)
        _, threshed_img = cv2.threshold(grey_img.copy(), 85, 255, cv2.THRESH_BINARY_INV)        # get the black line before edge detection
        threshed_img = cv2.erode(threshed_img.copy(), None, iterations = 2)
        threshed_img = cv2.dilate(threshed_img.copy(), None, iterations = 2)        # get rid of noises

        # cv2.Canny(file, kernelwidth, kernelheight)
        edged_img = cv2.Canny(threshed_img.copy(), 100, 100)     # find the edges using canny

        # cv2.HoughLinesP(file, distance resolution, angle resolution, threshold, minLineLength, maxLineGap)
        lines = cv2.HoughLinesP(edged_img, 1, np.pi/180, 20, minLineLength = 3)     # get the lines

        # render out all the linse on the raw_img
        try:
            if len(lines)>0:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(ROI_img, (x1,y1), (x2,y2), (0,255,0), 5)

        except TypeError as e:
            print('No lines found')
            print(e)

        mid_points = get_midpoint(lines)        # get the coordinates of the midpoints of lines
        if len(mid_points) > 0:
            for points in mid_points:
                cv2.circle(ROI_img, points, 1, (255,0,0))      # draw the midpoints

        angles, angles_ys = get_angles(lines)      # get the slope of all of the lines

        w_shifts, w_tans = get_weighted(mid_points, angles, angles_ys)      # get the weighted machines states

        output = np.concatenate((up_img, ROI_img.copy()), axis = 0)
        cv2.imshow('output', output)        # show the output image
        if cv2.waitKey(1) & 0xFF is ord('q'):       # waiting for the user to quit
            break

    cv2.destroyAllWindows()
    show_vote(w_shifts, w_tans)
    plt.show(block = True)

if __name__ == '__main__':
    main()      # run main
