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


def compute_errors_from_point_map(
    state,
    features: list,
    K: np.ndarray,
    week3
) -> tuple[float, float, dict]:
    """
    Builds a point-to-frame map and calculates the average reprojection error 
    for every individual point in the point cloud.
    
    Returns:
        mean_point_error: The overall average of all point errors.
        median_point_error: The overall median of all point errors.
        errors_by_point_id: A dictionary mapping point_id -> average_error in pixels.
    """
    
    # 1. Build the Point-to-Frame Map
    # Maps point_id -> list of tuples (image_id, keypoint_idx)
    point_to_frames = {}
    for (img_id, kp_idx), point_id in state.tracks.items():
        if img_id in state.registered_images:
            point_to_frames.setdefault(point_id, []).append((img_id, kp_idx))

    avg_error_per_point = {}

    # 2. Iterate through the map: evaluate every point against its frames
    for point_id, frames in point_to_frames.items():
        
        # Grab the 3D coordinate for this point
        X_3d = np.array([state.points3d[point_id]], dtype=np.float64)
        
        point_errors = []
        
        for img_id, kp_idx in frames:
            # Grab the camera pose for this frame
            R = state.camera_rotations[img_id]
            t = state.camera_translations[img_id]
            
            # Grab the 2D observation in this frame
            x_2d = np.array([features[img_id].keypoints[kp_idx].pt], dtype=np.float64)
            
            # Calculate the error for this specific point in this specific frame
            err = week3.compute_reprojection_errors(X_3d, x_2d, K, R, t)
            point_errors.append(err[0])
            
        # Calculate the average error for this specific 3D point across all its frames
        if point_errors:
            avg_error_per_point[point_id] = float(np.mean(point_errors))

    # 3. Calculate the global statistics for the entire point cloud
    if not avg_error_per_point:
        return 0.0, 0.0, {}

    all_point_averages = list(avg_error_per_point.values())
    point_mean = float(np.mean(all_point_averages))
    point_median = float(np.median(all_point_averages))

    return point_mean, point_median, avg_error_per_point