
import numpy as np
import re

def ang2rotmatrix(x, y, z, order='xyz'):
    """Compute the rotation matrix corresponding to three XYZ angles.
    
    Parameters
    ----------
    x, y, z : float
        The rotations along each axis given in radians.
    order : string
        The order in which rotations are applied. The possible values for
        this parameter are all permutations of 'x', 'y' and 'z' (e.g. 'xyz',
        'yxz'). The special value 'zxz' is also available (Euler angles).
        In such case, the three input angle parameters will be
        assumed to be z2, x, z, and the rotation order will be Z2 first,
        followed by X and Z.
    
    Returns
    -------
    R : 3x3 matrix
        The rotation matrix.
    """
    from numpy import array, dot
    
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    Rx = array([[1, 0, 0],
                [0, cx, -sx],
                [0, sx, cx]])
    Ry = array([[cy, 0, sy],
                [0, 1, 0],
                [-sy, 0, cy]])
    Rz = array([[cz, -sz, 0],
                [sz, cz, 0],
                [0, 0, 1]])
    
    if order == 'xyz':
        return dot(Rz, dot(Ry, Rx))
    elif order == 'xzy':
        return dot(Ry, dot(Rz, Rx))
    elif order == 'yxz':
        return dot(Rz, dot(Rx, Ry))
    elif order == 'yzx':
        return dot(Rx, dot(Rz, Ry))
    elif order == 'zxy':
        return dot(Ry, dot(Rx, Rz))
    elif order == 'zyx':
        return dot(Rx, dot(Ry, Rz))
    elif order == 'zxz':
        Rz2 = array([[cx, sx, 0],
                    [-sx, cx, 0],
                    [0, 0, 1]])
        return dot(Rz, dot(Ry, Rz2))
    else:
        raise ValueError, "Invalid order"

def sort_nicely( l ):
    """Sort the given list in the way that humans expect.
    
    Modified from nedbatchelder.com
    """ 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key=alphanum_key) 

def matarray(*args, **kwargs):
    """
    Construct a 2D array using a Matlab-like notation.
    You can specify the separator element using the keyword
    argument 'sep'. By default, it is 'None'. This is
    useful when creating block matrices.
    
    When the resulting matrix has only one element, matarray
    will return the element.
    
    Examples:
    >>> matarray(1,2,None,3,4)
    array([[1, 2],
           [3, 4]])
    >>> matarray(1, 2, '', 3, 4, sep='')
    array([[1, 2],
          [3, 4]])
    >>> R = np.ones((3,3))
    >>> t = np.zeros((3,1))
    >>> matarray(R, t, None, 0, 0, 0, 1)
    array([[ 1.,  1.,  1.,  0.],
           [ 1.,  1.,  1.,  0.],
           [ 1.,  1.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]])
    >>> matarray([3])
    3
    """
    sep = None
    arr = list(args)
    if len(arr) == 1:
        arr = arr[0]
    
    if not isinstance(arr, list):
        return arr
    
    if kwargs.has_key('sep'):
        sep = kwargs['sep']
    
    res = []
    aux = []
    for e in arr:
        if e == sep:
            # New row.
            if len(aux) > 0:
                res.append(np.hstack(aux))
            aux = []
        elif hasattr(e, '__iter__'):
            # Sub-matrix.
            submat = matarray(e, sep=sep)
            if submat is not None:
                aux.append(submat)
        else:
            aux.append(e)
    
    if len(aux) > 0:
        res.append(np.hstack(aux))
    if len(res) > 0:
        res = np.vstack(res)
        # If res has only one element, return the element.
        if res.size == 1:
            return res[0,0]
        return res
    
    return None

