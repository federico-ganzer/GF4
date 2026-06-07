from scipy.optimize import least_squares
import numpy as np
import cv2


def rodrigues_to_R(rvec):
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return R

def project_points(points3d, rvec, tvec, K):
    proj, _ = cv2.projectPoints(
        points3d.reshape(-1, 1, 3),
        rvec.reshape(3, 1),
        tvec.reshape(3, 1),
        K,
        None,
    )
    return proj.reshape(-1, 2)

def pack_params(camera_params, points3d):
    return np.hstack([camera_params.ravel(), points3d.ravel()])

def unpack_params(x, n_cams, n_pts):
    cam_block = x[:6 * n_cams].reshape(n_cams, 6)
    pt_block = x[6 * n_cams:].reshape(n_pts, 3)
    return cam_block, pt_block

def residuals_fn(x, n_cams, n_pts, camera_indices, point_indices, points_2d, K,
                 fixed_camera_mask, fixed_camera_params):
    cam_params, pts3d = unpack_params(x, n_cams, n_pts)

    cam_params[fixed_camera_mask] = fixed_camera_params[fixed_camera_mask]

    residuals = []
    for obs_id, (cam_idx, pt_idx) in enumerate(zip(camera_indices, point_indices)):
        rvec = cam_params[cam_idx, :3]
        tvec = cam_params[cam_idx, 3:]
        x_proj = project_points(pts3d[pt_idx:pt_idx+1], rvec, tvec, K)[0]
        residuals.extend(x_proj - points_2d[obs_id])
    return np.asarray(residuals, dtype=np.float64)


def least_squares_fit(camera_params0, points3d0, n_cams, n_pts, camera_indices, point_indices, points_2d, K, fixed_camera_mask, fixed_camera_params):
    
    x0 = pack_params(camera_params0, points3d0)
    
    result = least_squares(residuals_fn,
                           x0,
                           loss="huber",
                           f_scale=2.0,
                           method="trf",
                           args=(n_cams, n_pts, camera_indices, point_indices, points_2d, K, fixed_camera_mask, fixed_camera_params),
                           max_nfev=50,)
    
    return result
