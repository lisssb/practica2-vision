
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
from models import teapot
from models import cubo



def load_images(filename):
    filenames = glob.glob(filename + '/*_*.*')
    filenames = misc.sort_nicely(filenames)
    matriz = [imread(i) for i in filenames]
    return matriz

def play_ar(intrinsic, extrinsic, imgs, model):
    fig = ppl.gcf()
    fig.clf()

    v = model.vertices
    e = model.edges

    rotation_translation = misc.ang2rotmatrix(0, 0, 90)# get the rotation in z
    rotation_translation = np.column_stack((rotation_translation, [80, 80,0])) # add the traslation vecotr
    rotation_translation = np.row_stack((rotation_translation, [0,0,0,1]))

    for T, img in zip(extrinsic, imgs):

        fig.clf()
        # Do not show invalid detections.
        if T is None:
            continue
        T = np.dot(T, rotation_translation)

        # TODO: Project the model with proj.
        # Hint: T is the extrinsic matrix for the current image.
        points = proj(intrinsic, T, v)

        # TODO: Draw the model with plothom or plotedges.
        plothom(points)

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
    num_points = chessboard_shape[0]*chessboard_shape[1]
    points = np.ndarray(shape=(num_points,3))
    coordinate_x = 0
    for i in range(chessboard_shape[1]):
        coordinate_y = 0
        for j in range(chessboard_shape[0]):
            indice_i = i*chessboard_shape[0]+j
            points[indice_i][0] = coordinate_x
            points[indice_i][1] = coordinate_y
            points[indice_i][2] = 0
            coordinate_y = coordinate_y+dy
        coordinate_x = coordinate_x+dx
    return points

def calculate_diagonal_fov(intrinsic, image_size):
    """calculate_diagonal_fov.
    This function determines the diagonal angle of view that encompasses the camera.
    Parameters
    ----------
    intrinsic : ndarray
        3x3 intrinsic matrix
    image_size : tuple
        Size (height,width) of the images captured by the camera.

    Output
    ------
    Angle : integer
        angle in degrees
    """
    focal_x = intrinsic[0][0]
    focal_y = intrinsic[1][1]
    W = image_size[1]
    H = image_size[0]
    radians = 2 * math.atan(math.sqrt((W/(2*focal_x))**2 + (H/(2*focal_y))**2))
    return radians * 180 / math.pi

'''
funcion que convierte los puntos dados en verts 3d a puntos 2d en la image_points'''
def proj(K, T, verts):
    points_list=[]
    rotation_translation = np.delete(T,3,0) # Eliminamos la ltima fila que no nos aporta informaic
    for v in verts.T:
        rt = np.dot(rotation_translation, v)
        vert = np.dot( K , rt)
        points_list.append(vert)

    points = np.asarray(points_list)
    return points.T

def plothom(points):
    x = points[0]
    y = points[1]
    landa = points[2]
    ppl.plot(x/landa, y/landa, marker='.', linestyle='.', color='#9e56eb')#'#009999')
    ppl.show()

def get_distance(extrinsic, extrinsic_r):
    right_camera_in_left = np.dot(extrinsic[1], extrinsic_r[1][:,3])
    distance = right_camera_in_left - extrinsic[1][:,3]
    return right_camera_in_left, np.linalg.norm(distance)

def main():
    images =  load_images('left')
    corners = [cv2.findChessboardCorners(i, (8,6)) for i in images]
    imgs2 = copy.deepcopy(images)
    for im, cor in zip(imgs2, corners):
        if(cor[0]):
            cv2.drawChessboardCorners(im, (8,6), cor[1], cor[0])

    #This for is used to draw all the square on each image
    # for i in imgs2:
    #         ppl.imshow(i)
    #         ppl.show()

    size = images[0].shape[0:2]
    intrinsic, extrinsic, dist_coeff = calibrate(corners, get_chessboard_points((8, 6), 30,30), size)
    np.savez('calib_left', intrinsic=intrinsic, extrinsic=extrinsic)

    # play_ar(intrinsic, extrinsic, images, teapot)
    # play_ar(intrinsic, extrinsic, images, bunny)
    # play_ar(intrinsic, extrinsic, images, cubo)

    images_r =  load_images('right')
    corners_r = [cv2.findChessboardCorners(i, (8,6)) for i in images_r]
    size_r = images_r[0].shape[0:2]
    intrinsic_r, extrinsic_r, dist_coeff_r = calibrate(corners_r, get_chessboard_points((8, 6), 30,30), size_r)
    np.savez('calib_right', intrinsic=intrinsic_r, extrinsic=extrinsic_r)


    # print calculate_diagonal_fov(intrinsic, size)
    #problema 12
    right_camera_in_left, distance = get_distance(extrinsic, extrinsic_r)
    print 'right camera in left ::\n', right_camera_in_left
    print 'dsistance ::\n', distance
main()
