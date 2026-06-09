import numpy as np


def compute_global_reprojection_error(
    state,
    features: list,
    K: np.ndarray,
    week3
) -> tuple[float, float]:
    """
    Computes the mean and median reprojection error across all registered 
    cameras and all 3D points they observe.
    """
    # Group observations by image to avoid O(N^2) dictionary scanning
    obs_by_image = {}
    for (img_id, kp_idx), point_id in state.tracks.items():
        if img_id in state.registered_images:
            obs_by_image.setdefault(img_id, []).append((kp_idx, point_id))

    all_errors = []

    for img_id, obs in obs_by_image.items():
        R = state.camera_rotations[img_id]
        t = state.camera_translations[img_id]

        # Unpack the indices for this specific image
        kp_indices = [kp for kp, _ in obs]
        pt_indices = [pt for _, pt in obs]

        # Fetch the corresponding 3D points and 2D keypoint coordinates
        points3d = state.points3d[pt_indices]
        pts2d = np.array([features[img_id].keypoints[idx].pt for idx in kp_indices], dtype=np.float64)

        errors = week3.compute_reprojection_errors(points3d, pts2d, K, R, t)
        all_errors.append(errors)

    if not all_errors:
        return 0.0, 0.0

    # Flatten the list of arrays into a single 1D array of all errors
    all_errors_flat = np.concatenate(all_errors)
    
    return float(np.mean(all_errors_flat)), float(np.median(all_errors_flat))


def compute_pointwise_reprojection_error(
    state,
    features: list,
    K: np.ndarray,
    week3
) -> tuple[float, float]:
    """
    Computes the mean and median of the per-point average reprojection errors.
    """
    # Group by image first for fast, batched 3D-to-2D projection
    obs_by_image = {}
    for (img_id, kp_idx), point_id in state.tracks.items():
        if img_id in state.registered_images:
            obs_by_image.setdefault(img_id, []).append((kp_idx, point_id))

    # Dictionary to collect all errors for each specific 3D point
    errors_by_point = {}

    for img_id, obs in obs_by_image.items():
        R = state.camera_rotations[img_id]
        t = state.camera_translations[img_id]

        kp_indices = [kp for kp, _ in obs]
        pt_indices = [pt for _, pt in obs]

        points3d = state.points3d[pt_indices]
        pts2d = np.array([features[img_id].keypoints[idx].pt for idx in kp_indices], dtype=np.float64)

        # Calculate errors for this batch
        errors = week3.compute_reprojection_errors(points3d, pts2d, K, R, t)

        # Distribute the calculated errors to their respective 3D points
        for pt_id, err in zip(pt_indices, errors):
            errors_by_point.setdefault(pt_id, []).append(err)

    if not errors_by_point:
        return 0.0, 0.0

    # 1. Calculate the average error for EACH point
    avg_error_per_point = [np.mean(errs) for errs in errors_by_point.values()]

    # 2. Calculate the global mean and median of those point-averages
    return float(np.mean(avg_error_per_point)), float(np.median(avg_error_per_point))
