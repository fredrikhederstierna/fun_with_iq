
# Simple interactive image quality enhancement utility.
# Fredrik Hederstierna 2022

import numpy as np
import cv2

import matplotlib
matplotlib.use('TkAgg')

from matplotlib import pyplot as plt

win_name = 'main'

gamma_value = 1.0
clahe_value = 1.0
tint_value  = 1.0

global fig

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
		      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def draw_window():
    global img_file
    img_copy = img_file.copy()
    img_copy_orig = img_copy.copy()

    # GAMMA
    img_gamma = adjust_gamma(img_copy, gamma=gamma_value)
    img_gamma_copy = img_gamma.copy()

    cv2.putText(img_gamma_copy,
                'gamma=' + "{:.2f}".format(gamma_value),
                (0, img_gamma_copy.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 3)

    img_gamma_stacked = np.hstack((img_copy, img_gamma_copy))

    # apply CLAHE on orig copy image
    lab        = cv2.cvtColor(img_copy_orig, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe      = cv2.createCLAHE(clipLimit=clahe_value, tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab        = cv2.merge(lab_planes)
    img_clahe_orig = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    cv2.putText(img_clahe_orig,
                'clip=' + "{:.2f}".format(clahe_value),
                (0, img_clahe_orig.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 3)
    
    # apply CLAHE on GAMMA copy image
    lab        = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe      = cv2.createCLAHE(clipLimit=clahe_value, tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab        = cv2.merge(lab_planes)
    img_clahe_gamma = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)



    # apply TINT, modify S (Saturation) in HSV image
    lab     = cv2.cvtColor(img_clahe_gamma, cv2.COLOR_BGR2HSV)
    (H, S, V) = cv2.split(lab)
    # Obtain array of square of each element in x
    global tint_value
    for row in range(S.shape[1]):
        for col in range(S.shape[0]):
            s = S[col][row] * tint_value
            if s > 255:
                s = 255
            S[col][row] = s
    lab = cv2.merge([H, S, V])
    img_clahe_gamma_tint = cv2.cvtColor(lab, cv2.COLOR_HSV2BGR)
    # USE THIS IMAGE, ADDING TINT ASWELL
    img_clahe_gamma = img_clahe_gamma_tint



    ########## HISTOGRAM
    img_yuv = cv2.cvtColor(img_clahe_gamma, cv2.COLOR_BGR2YUV)
    y_hist = cv2.calcHist([img_yuv],[0],None,[256],[0,256])
    global fig
    fig.clear(True)
    #plt.imshow(img_hist) #, interpolation='nearest')
    plt.plot(y_hist, color='b')
    plt.title('Histogram Y channel')
    plt.grid(color='gray', linestyle='--', linewidth=1)
    plt.xlim([0,256])
    # redraw the canvas
    fig.canvas.draw()
    # convert canvas to image
    #imgx = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    imgx = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    imgx  = imgx.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # img is rgb, convert to opencv's default bgr
    imgx = cv2.cvtColor(imgx, cv2.COLOR_RGB2BGR)
    # display image with opencv or any operation you like
    cv2.imshow("Histogram over Y component", imgx)


    
    cv2.putText(img_clahe_gamma,
                'gamma=' + "{:.2f}".format(gamma_value),
                (0, img_clahe_gamma.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 3)
    cv2.putText(img_clahe_gamma,
                'clip=' + "{:.2f}".format(clahe_value),
                (0, img_clahe_gamma.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 3)
    cv2.putText(img_clahe_gamma,
                'tint=' + "{:.2f}".format(tint_value),
                (300, img_clahe_gamma.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 3)
    
    # stacking images side-by-side (hstack / vstack)
    img_clahe_stacked = np.hstack((img_clahe_orig, img_clahe_gamma))

    img_all = np.vstack((img_gamma_stacked, img_clahe_stacked))

    
    # DRAW IT ALL
    global win_name
    cv2.imshow(win_name, img_all)


def on_change_stretch(value):
    global gamma_value
    gamma_value = (0.01 + value/50.0)
    draw_window()

def on_change_equalize(value):
    global clahe_value
    clahe_value = (0.01 + value/5.0)
    draw_window()

def on_change_tint(value):
    global tint_value
    tint_value = 2.0 * (value / 100.0)
    draw_window()

#img_filename = r'./table3.png'
#img_filename = r'./testx.jpg'
#img_filename = r'./badtree.jpeg'
img_filename = r'./badtree2.jpeg'
#img_filename = r'./tree4.jpg'
#img_filename = r'./tree5.jpeg'
#img_filename = r'./apples.jpeg'
#img_filename = r'./beans.png'
img_file = cv2.imread(img_filename)
img_file = resize_with_aspect_ratio(img_file, width=480)

# Note that the previous function call will return the image as a numpy ndarray.

fig = plt.figure()

#win_name = 'test'
#cv2.imshow(win_name, img_file)
draw_window()

cv2.createTrackbar('brightness', win_name, 0, 100, on_change_stretch)
cv2.setTrackbarPos('brightness', win_name, 50)
cv2.createTrackbar('contrast',   win_name, 0, 100, on_change_equalize)
cv2.setTrackbarPos('contrast',   win_name, 0)
cv2.createTrackbar('color',      win_name, 0, 100, on_change_tint)
cv2.setTrackbarPos('color',      win_name, 50)

cv2.waitKey(0)
cv2.destroyAllWindows()

# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#	help="path to input image")
#args = vars(ap.parse_args())
# load the original image
#original = cv2.imread(args["image"])
