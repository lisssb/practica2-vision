
import time
import cv2
import misc
import glob
import copy
import numpy as np
from operator import itemgetter
from scipy.misc import imread
import matplotlib.pyplot as ppl
from matplotlib.pyplot import imshow, plot
# Una de las chapuzas de OpenCV ...
from cv import CV_CALIB_FIX_ASPECT_RATIO


def play_ar(intrinsic, extrinsic, imgs, model):
    
    fig = ppl.gcf()
    fig.clf()
    
    v = model.vertices
    e = model.edges
    
    for T, img in zip(extrinsic, imgs):
        fig.clf()
        
        # Do not show invalid detections.
        if T is None:
            continue
        
        # TODO: Project the model with proj.
        # Hint: T is the extrinsic matrix for the current image.


        # TODO: Draw the model with plothom or plotedges.



        # Plot the image.
        imshow(img)
        ppl.draw()
        time.sleep(0.1)

def calibrate(image_corners, chessboard_points, image_size):
    """Calibrate a camera.

    This function determines the intrinsic matrix and the extrinsic
    matrices of a camera.

    Parameters
    ----------
    image_corners : list
        List of the M outputs of cv2.findChessboardCorners, where
        M is the number of images.
    chessboard_points : ndarray
        Nx3 matrix with the (X,Y,Z) world coordinates of the
        N corners of the calibration chessboard pattern.
    image_size : tuple
        Size (height,width) of the images captured by the camera.

    Output
    ------
    intrinsic : ndarray
        3x3 intrinsic matrix
    dist_coefs: Non-linear distortion coefficients for the camera
    """
    valid_corners = [carr for validP,carr in image_corners if validP]
    num_images = len(image_corners)
    num_valid_images = len(valid_corners)
    num_corners = len(valid_corners[0][1])

    # Input data.
    object_points = np.array([chessboard_points] * num_valid_images, dtype=np.float32)
    image_points = np.array([carr[:,0,:] for carr in valid_corners], dtype=np.float32)
    # Output matrices.
    intrinsics = np.identity(3)
    dist_coeffs = np.zeros(4)

    # Calibrate for square pixels
    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, intrinsics, dist_coeffs, flags=CV_CALIB_FIX_ASPECT_RATIO)

    def vecs2matrices((rvec,tvec)):
        R,jacov = cv2.Rodrigues(rvec)
        return misc.matarray(R,tvec,None,0,0,0,1)

    extrinsicsAux = map(vecs2matrices,zip(rvecs,tvecs))

    extrinsicsiter=iter(extrinsicsAux)
    extrinsics=[]
    for i,corner in enumerate(image_corners):
        if corner[0]:
            extrinsics.append(extrinsicsiter.next())
        else:
            extrinsics.append(None)

    return intrinsics, extrinsics, dist_coeffs


