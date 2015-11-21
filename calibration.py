
import time
import cv2
import misc
import glob
import copy
import numpy as np
from operator import itemgetter
from scipy.misc import imread
# from scipy.misc import sort_nicely
import matplotlib.pyplot as ppl
from matplotlib.pyplot import imshow, plot
# Una de las chapuzas de OpenCV ...
from cv import CV_CALIB_FIX_ASPECT_RATIO
import math
from models import bunny



def load_images(filename):
    filenames = glob.glob(filename + '/left_*.*')
    filenames = misc.sort_nicely(filenames)
    matriz = [imread(i) for i in filenames]
    return matriz


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

def get_chessboard_points(chessboard_shape, dx, dy):
    num_points=chessboard_shape[0]*chessboard_shape[1]
    points=np.ndarray(shape=(num_points,3))
    acum_x=0
    for i in range(chessboard_shape[0]):
        acum_y=0
        for j in range(chessboard_shape[1]):
            idx = i*chessboard_shape[1]+j
            points[idx][0]=acum_x
            points[idx][1]=acum_y
            points[idx][2]=0
            acum_y=acum_y+dy
        acum_x=acum_x+dx
    return points

def calculate_fov(intrinsic, image_size):
    W = image_size[1]
    focal_length_px = intrinsic[0][0]
    mid_fov = math.atan(W/(2*focal_length_px))
    return mid_fov * 2

'''
funcion que convierte los puntos dados en verts 3d a puntos 2d en la image_points'''
def proj(K, T, verts):
    screen_points_list=[]
    rotation_translation = np.delete(T,3,0) # Eliminamos la ltima fila que no nos aporta informaic
    for v in verts.T:
        aux = np.dot(rotation_translation, v)
        vert = np.dot( K , aux)
        screen_points_list.append(vert)

    screen_points = np.asarray(screen_points_list)
    return screen_points.T

def main():
    images =  load_images('left')
    corners = [cv2.findChessboardCorners(i, (8,6)) for i in images]
    imgs2 = copy.deepcopy(images)
    i = 0
    for im, cor in zip(imgs2, corners):
        if(cor[0]):
            cv2.drawChessboardCorners(im, (8,6), cor[1], cor[0])

    # for i in imgs2:
    #         ppl.imshow(i)
    #         ppl.show()

    size = images[0].shape[0:2]
    intrinsic, extrinsic, dist_coeff = calibrate(corners, get_chessboard_points((8, 6), 300,300), size)
    np.savez('calib_left', intrinsic=intrinsic, extrinsic=extrinsic)
    # print 'focal value'
    # print calculate_fov(intrinsic, size)

    proj(intrinsic, extrinsic, bunny.vertices)

main()
