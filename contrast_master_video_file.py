
# Simple interactive image quality enhancement utility.
# Added video as input source.
# Fredrik Hederstierna 2022/2023

# python3 -m pip uninstall opencv-python-headless
# python3 -m pip install --upgrade opencv-python

# Some info on meta in the video stream
#
#  stream_index   = frame count
#  key_frame      = iframe or pframe
#  pkt_pts        = packet presentation time stamp
#  pkt_pts_time   = packet presentation time stamp time
#  pkt_duration_time = frame rate
#  pkt_pos        = packet position (relative to the audio packets to keep sync with video)
#  pkt_size       = iframes are much larger and pframes are generally much smaller

import numpy as np
import cv2

import matplotlib
matplotlib.use('TkAgg')

from matplotlib import pyplot as plt

win_name = 'main'

gamma_value = 1.0
clahe_value = 1.0
tint_value  = 1.0
noise_value = 1.0

frame_count = 0
frame_count_max = 0

global fig

# resize image, keeping the aspect ratio
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

    # NOISE
    # Bi-lateral (SIGMA filter)
    # https://lindevs.com/bilateral-filtering-of-the-image-using-opencv
    #
    # The d parameter defines filter size. In other words, it is the diameter of each pixel neighborhood.
    # Sigma in the color space and sigma in the coordinate space control the amount of filtering.
    #img_noise = cv2.bilateralFilter(img_copy, d=5, sigmaColor=50, sigmaSpace=50)
    img_noise = cv2.bilateralFilter(img_copy, d=int(noise_value), sigmaColor=50, sigmaSpace=50)
    img_noise_copy = img_noise.copy()

    # add labels overlay
    cv2.putText(img_noise_copy,
                'sigma=' + "{:d}".format(int(noise_value)),
                (0, img_noise_copy.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 3)

    # GAMMA on NOISE image
    #img_gamma = adjust_gamma(img_copy, gamma=gamma_value)
    img_gamma = adjust_gamma(img_noise, gamma=gamma_value)
    img_gamma_copy = img_gamma.copy()

    # add labels overlay
    cv2.putText(img_gamma_copy,
                'sigma=' + "{:d}".format(int(noise_value)),
                (0, img_gamma_copy.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 3)
    # add labels overlay
    cv2.putText(img_gamma_copy,
                'gamma=' + "{:.2f}".format(gamma_value),
                (0, img_gamma_copy.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 3)

    img_gamma_stacked = np.hstack((img_copy, img_noise_copy, img_gamma_copy))

    # apply CLAHE on orig copy image
    lab        = cv2.cvtColor(img_copy_orig, cv2.COLOR_BGR2LAB)
    lab_planes_tuple = cv2.split(lab)
    lab_planes_list = list(lab_planes_tuple)
    clahe      = cv2.createCLAHE(clipLimit=clahe_value, tileGridSize=(8,8))
    lab_planes_list[0] = clahe.apply(lab_planes_list[0])
    lab_planes_tuple = tuple(lab_planes_list)
    lab        = cv2.merge(lab_planes_tuple)
    img_clahe_orig = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # add labels overlay
    cv2.putText(img_clahe_orig,
                'clip=' + "{:.2f}".format(clahe_value),
                (0, img_clahe_orig.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 3)
    
    # apply CLAHE on GAMMA+NOISE copy image
    lab        = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2LAB)
    lab_planes_tuple = cv2.split(lab)
    lab_planes_list = list(lab_planes_tuple)
    clahe      = cv2.createCLAHE(clipLimit=clahe_value, tileGridSize=(8,8))
    lab_planes_list[0] = clahe.apply(lab_planes_list[0])
    lab_planes_tuple = tuple(lab_planes_list)
    lab        = cv2.merge(lab_planes_tuple)
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

    
    # add labels overlay
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
    cv2.putText(img_clahe_gamma,
                'sigma=' + "{:d}".format(int(noise_value)),
                (300, img_clahe_gamma.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 3)

    img_clahe_noise = img_clahe_orig

    # stacking images side-by-side (hstack / vstack)
    img_clahe_stacked = np.hstack((img_clahe_orig, img_clahe_noise, img_clahe_gamma))

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

def on_change_noise(value):
    global noise_value
    noise_value = (25.0 * (value / 100.0)) + 1
    draw_window()




# define a video capture object

vid_filename = r'./day_to_night.mp4'
vid = cv2.VideoCapture(vid_filename)
#
# Debug meta info
fps = vid.get(cv2.CAP_PROP_FPS)
nbr = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
dur = nbr/fps
ts  = vid.get(cv2.CAP_PROP_POS_MSEC)
print('Fps %d'%fps, 'Frames %d'%nbr, 'Duration %d seconds'%dur, 'First timestamp %d'%ts)

first_round = True
frame_count_max = nbr

fig = plt.figure()

while(vid.isOpened()):

    # Capture the video frame
    # by frame
    frame_exists, cur_frame = vid.read()

    if frame_exists == False:
        break;

    frame_count += 1

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    #img_filename = r'./noise7.jpeg'
    #img_file = cv2.imread(img_filename)
    img_file = resize_with_aspect_ratio(cur_frame, width=480)

    # Note that the previous function call will return the image as a numpy ndarray.

    #win_name = 'test'
    #cv2.imshow(win_name, img_file)
    draw_window()

    if (first_round):
        cv2.createTrackbar('brightness', win_name, 0, 100, on_change_stretch)
        cv2.setTrackbarPos('brightness', win_name, 50)
        cv2.createTrackbar('contrast',   win_name, 0, 100, on_change_equalize)
        cv2.setTrackbarPos('contrast',   win_name, 0)
        cv2.createTrackbar('color',      win_name, 0, 100, on_change_tint)
        cv2.setTrackbarPos('color',      win_name, 50)
        cv2.createTrackbar('noise',      win_name, 0, 100, on_change_noise)
        cv2.setTrackbarPos('noise',      win_name, 0)
        first_round = False

    ts = vid.get(cv2.CAP_PROP_POS_MSEC)
    cv2.setWindowTitle(win_name, "Frame %d/%d Time %d/%d ms"%(frame_count, frame_count_max, ts, dur*1000))

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#	help="path to input image")
#args = vars(ap.parse_args())
# load the original image
#original = cv2.imread(args["image"])

