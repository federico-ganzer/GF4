from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import numpy as np
import cv2


def rodrigues_to_R(rvec):
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return R

def project_points_batch(points3d, rvec, tvec, K):
    """Project a batch of 3D points using one camera pose."""
    R = rodrigues_to_R(rvec)
    X_cam = R @ points3d.T + tvec.reshape(3, 1)
    x = K @ X_cam
    x = x[:2] / x[2]
    return x.T

def pack_params(camera_params, points3d):
    return np.hstack([camera_params.ravel(), points3d.ravel()])

def unpack_params(x, n_cams, n_pts):
    cam_block = x[:6 * n_cams].reshape(n_cams, 6)
    pt_block = x[6 * n_cams:].reshape(n_pts, 3)
    return cam_block, pt_block

def residuals_fn(
    x,
    n_cams,
    n_pts,
    camera_indices,
    point_indices,
    points_2d,
    K,
    fixed_camera_mask,
    fixed_camera_params,
):
    cam_params, pts3d = unpack_params(x, n_cams, n_pts)
    cam_params[fixed_camera_mask] = fixed_camera_params[fixed_camera_mask]

    residuals = np.empty((len(camera_indices), 2), dtype=np.float64)

    unique_cams, inverse = np.unique(camera_indices, return_inverse=True)
    for cam_id in unique_cams:
        cam_mask = camera_indices == cam_id
        local_obs = np.nonzero(cam_mask)[0]

        rvec = cam_params[cam_id, :3]
        tvec = cam_params[cam_id, 3:]
        pts = pts3d[point_indices[cam_mask]]

        projected = project_points_batch(pts, rvec, tvec, K)
        residuals[cam_mask] = projected - points_2d[cam_mask]

    return residuals.ravel()

def build_jac_sparsity(n_cams, n_pts, camera_indices, point_indices):
    m = len(camera_indices) * 2
    n = 6 * n_cams + 3 * n_pts
    A = lil_matrix((m, n), dtype=bool)

    for obs_id, (cam_idx, pt_idx) in enumerate(zip(camera_indices, point_indices)):
        row = 2 * obs_id
        cam_start = cam_idx * 6
        pt_start = 6 * n_cams + pt_idx * 3
        A[row:row+2, cam_start:cam_start+6] = True
        A[row:row+2, pt_start:pt_start+3] = True

    return A

def least_squares_fit(camera_params0, points3d0, n_cams, n_pts, camera_indices, point_indices, points_2d, K, fixed_camera_mask, fixed_camera_params):
    
    x0 = pack_params(camera_params0, points3d0)
    sparsity = build_jac_sparsity(n_cams, n_pts, camera_indices, point_indices)
    result = least_squares(residuals_fn,
                           x0,
                           loss="huber",
                           f_scale=2.0,
                           method="trf",
                           jac_sparsity=sparsity,
                           args=(n_cams, n_pts, camera_indices, point_indices, points_2d, K, fixed_camera_mask, fixed_camera_params),
                           max_nfev=50,)
    
    return result
