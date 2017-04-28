import numpy as np
import contextlib
import sys

import math

from menpo.transform import UniformScale, Translation, Homogeneous, scale_about_centre, Rotation, Scale
from menpo.shape import PointCloud, TriMesh
from functools import reduce


class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    f = open('/dev/null', 'w')
    
    save_stdout = sys.stdout
    sys.stdout = f
    yield
    sys.stdout = save_stdout

def clip_texture(image):
    image.pixels = image.pixels.clip(0, 1)
    return image
    
def rescale_values(arr, minimum, maximum):
    min_, max_ = arr.min(), arr.max()
    sf = ((maximum - minimum) * 1.0) / (max_ - min_)
    return ((arr - min_) * sf) + minimum

def clip_to_image(height, width): 
    # 2. invert the y direction (up becomes down)
    invert_y = Scale([1, -1])
    # 3. [-1, 1] [-1, 1] -> [0, 2] [0, 2]
    t = Translation([1, 1])
    # 4. [0, 2] [0, 2] -> [0, 1] [0, 1]
    unit_scale = Scale(0.5, n_dims=2)
    # 5. [0, 1] [0, 1] -> [0, w] [0, h]
    im_scale = Scale([width, height])
    # 6. [0, w] [0, h] -> [0, h] [0, w]
    xy_yx = Homogeneous(np.array([[0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]], dtype=np.float))
    # reduce the full transform chain to a single affine matrix
    transforms = [invert_y, t, unit_scale, im_scale, xy_yx]
    return reduce(lambda a, b: a.compose_before(b), transforms)

def get_vertices(mesh):
    old_to_new = mesh.trilist.ravel()
    new_trilist = np.arange(old_to_new.shape[0]).reshape([-1, 3])
    new_points = mesh.points[old_to_new]
    return mesh.points

def calculate_weights(x):
    return model.project(AlignmentSimilarity(x, model.mean()).apply(x))

def model_to_clip_space(im, mesh):
    view_t, c_t, proj_t = retrieve_camera_matrix(im, mesh, initialize=True)
    proj_t = weak_projection_matrix(im.shape[1], im.shape[0], view_t.apply(mesh))
    view_t = view_t.compose_before(axes_flip_t)
    transform = Homogeneous(proj_t.dot(view_t.h_matrix))
    return transform

def duplicate_vertices(mesh):
    # generate a new mesh with unique vertices per triangle
    # (i.e. duplicate verts so that each triangle is unique)    old_to_new = mesh.trilist.ravel()
    old_to_new = mesh.trilist.ravel()
    new_trilist = np.arange(old_to_new.shape[0]).reshape([-1, 3])
    new_points = mesh.points[old_to_new]
    return TriMesh(new_points, trilist=new_trilist), old_to_new

def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix."""

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / np.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        try:
            w, V = np.linalg.eigh(K)
        except:
            return quaternion_from_matrix(matrix, isprecise=True)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q

def quaternion_matrix(quaternion, eps=.00000001):
    """Return homogeneous rotation matrix from quaternion."""
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < eps:
        return np.identity(4)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])

def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = np.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = np.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az

def retrieve_camera_matrix(image, mesh, group=None, initialize=True):
    import cv2
    
    drop_h = Homogeneous(np.eye(4)[:3])
    flip_xy_yx = Homogeneous(np.array([[0, 1, 0],
                                       [1, 0, 0],
                                       [0, 0, 1]]))

    rows = image.shape[0]
    cols = image.shape[1]
    max_d = max(rows, cols)
    camera_matrix = np.array([[max_d, 0,     cols / 2.0],
                              [0,     max_d, rows / 2.0],
                              [0,     0,     1.0]])
    distortion_coeffs = np.zeros(4)

    # Initial guess for rotation/translation.
    if initialize:
        r_vec = np.array([[-2.7193267 ], [-0.14545351], [-0.34661788]])
        t_vec = np.array([[0.], [ 0. ], [280.]])
        converged, r_vec, t_vec = cv2.solvePnP(mesh.landmarks[group].lms.points, 
                                               image.landmarks[group].lms.points[:, ::-1], 
                                               camera_matrix, 
                                               distortion_coeffs, r_vec, t_vec, 1)
    else:
        converged, r_vec, t_vec = cv2.solvePnP(mesh.landmarks[group].lms.points, 
                                               image.landmarks[group].lms.points[:, ::-1], 
                                               camera_matrix, 
                                               distortion_coeffs)

    rotation_matrix = cv2.Rodrigues(r_vec)[0]
    
    h_camera_matrix = np.eye(4)
    h_camera_matrix[:3, :3] = camera_matrix

    t_vec = t_vec.ravel()

    if t_vec[2] < 0:
        print('Position has a negative value in z-axis')

    c = Homogeneous(h_camera_matrix)
    t = Translation(t_vec)
    r = Rotation(rotation_matrix)

    view_t = r.compose_before(t)
    proj_t = c.compose_before(drop_h).compose_before(flip_xy_yx)
    return view_t, c, proj_t


def weak_projection_matrix(width, height, mesh_camera_space):

    # Identify how far and near the mesh is in camera space.
    # we want to ensure that the near and far planes are
    # set so that all the mesh is displayed.
    near_bounds, far_bounds = mesh_camera_space.bounds()

    # Rather than just use the bounds, we add 10% in each direction
    # just to avoid any numerical errors.
    average_plane = (near_bounds[-1] + far_bounds[-1]) * 0.5
    padded_range = mesh_camera_space.range()[-1] * 1.1
    near_plane = average_plane - padded_range
    far_plane = average_plane + padded_range

    plane_sum = far_plane + near_plane
    plane_prod = far_plane * near_plane
    denom = far_plane - near_plane
    max_d = max(width, height)

    return np.array([[2.0 * max_d / width, 0,                    0,                    0],
                     [0,                   2.0 * max_d / height, 0,                    0],
                     [0,                   0,                    (-plane_sum) / denom, (-2.0 * plane_prod) / denom],
                     [0,                   0,                    -1,                   0]])


def duplicate_vertices(mesh):
    # generate a new mesh with unique vertices per triangle
    # (i.e. duplicate verts so that each triangle is unique)    old_to_new = mesh.trilist.ravel()
    old_to_new = mesh.trilist.ravel()
    new_trilist = np.arange(old_to_new.shape[0]).reshape([-1, 3])
    new_points = mesh.points[old_to_new]
    return TriMesh(new_points, trilist=new_trilist), old_to_new


def crop_face(img, boundary=50, group=None, shape=(256, 256), order=1):
    pc = img.landmarks[group].lms
    nan_points = np.any(np.isnan(pc.points).reshape(-1, 2), 1)

    pc = PointCloud(pc.points[~nan_points, :])
    min_indices, max_indices = pc.bounds(boundary=boundary)
    h = max_indices[0] - min_indices[0]
    w = max_indices[1] - min_indices[1]
    pad = abs(w - h)

    try:
        index = 1 - int(w > h)
        min_indices[index] -= int(pad / 2.)
        max_indices[index] += int(pad / 2.) + int(pad) % 2

        img = img.crop(min_indices, max_indices, constrain_to_boundary=True)
    except Exception as e:
        print("Exception in crop_face", e)

    img = img.resize(shape, order=order)
    return img
