from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
import sys
import cv2

import numpy as np

import time

'''from week3.two_view_utils import (
    ThirdViewResult,
    TwoViewResult,
    build_2d3d_correspondences,
    compute_depths,
    compute_reprojection_errors,
    draw_reprojection_overlay,
    draw_single_image_reprojection_overlay,
    ensure_dir,
    estimate_essential_matrix,
    estimate_camera_pose_pnp,
    filter_reconstructed_points,
    make_camera_matrix,
    plot_multi_view_reconstruction,
    plot_patch_cloud_reconstruction,
    plot_two_view_reconstruction,
    recover_relative_pose,
    sample_point_colours,
    save_csv,
    triangulate_points,
    write_ply,
)'''

DEFAULT_WEEK2_DIR = Path(__file__).resolve().parents[1] / "week2"
DEFAULT_WEEK3_DIR = Path(__file__).resolve().parents[1] / "week3"


@dataclass
class PairwiseEdge:
    image_i: int
    image_j: int
    matches: list
    inlier_mask: np.ndarray
    inlier_count: int
    essential_matrix: np.ndarray


@dataclass
class ReconstructionState:
    registered_images: list[int]
    unregistered_images: list[int]
    camera_rotations: dict[int, np.ndarray]
    camera_translations: dict[int, np.ndarray]
    tracks: dict[tuple[int, int], int]
    points3d: np.ndarray
    point_colors: np.ndarray
    
    
import open3d as o3d

class ReconstructionPlotter:
    def __init__(self, window_name= "Live Reconstruction"):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name)
        self.point_cloud = None
        self.cameras = []
        self.added = False
        
    def update(self, state: ReconstructionState, K: np.ndarray):
        if self.added:
            self.vis.clear_geometries() # clear existing reconstruction to update
            
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(state.points3d)
        self.point_cloud.colors = o3d.utility.Vector3dVector(state.point_colors.astype(np.float64)/255)
        self.vis.add_geometry(self.point_cloud)

        for img_id in state.registered_images:
            R = state.camera_rotations[img_id]
            t = state.camera_translations[img_id]
            frustum = self._build_camera_frustum(R, t, scale=0.1)
            self.vis.add_geometry(frustum)
        
        self.vis.poll_events()
        self.vis.update_renderer()
        self.added = True # marks existing reconstruction
    
    def _build_camera_frustum(self, R: np.ndarray, t: np.ndarray, scale: float):
        R_cw = R.T # convert to camera2world view
        center = -R_cw @ t.reshape(3,1)
        
        # Simple pyramid corners in camera space
        corners = scale * np.array([
            [-1, -0.75, 2], [1, -0.75, 2],
            [1, 0.75, 2], [-1, 0.75, 2],
        ], dtype=np.float64)
        
        corners_world = (R_cw @ corners.T).T + center.ravel()
        
        lines = [[0, 1], [1, 2], [2, 3], [3, 0],
                 [0, 4], [1, 4], [2, 4], [3, 4]]
        colors = [[1, 0, 0] for _ in lines]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(
            np.vstack([corners_world, center.ravel()])
        )
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set
    
    def close(self):
        self.vis.destroy_window()

def load_week2_module(week2_dir: Path):
    module_path = Path(week2_dir) / "sfm_utils.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find Week 2 sfm_utils.py: {module_path}")

    spec = importlib.util.spec_from_file_location("week2_sfm_utils", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Week 2 module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_week3_module(week3_dir: Path):
    """Load the completed Week 3 two_view_utils.py by path."""
    module_path = Path(week3_dir) / "two_view_utils.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find Week 3 two_view_utils.py: {module_path}")

    spec = importlib.util.spec_from_file_location("week3_two_view_utils", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Week 3 module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GF4 Week 4 incremental sparse reconstruction pipeline."
    )
    image_group = parser.add_mutually_exclusive_group(required=True)
    image_group.add_argument(
        "--images",
        nargs="+",
        type=Path,
        help="Paths to all images in the sequence.",
    )
    image_group.add_argument(
        "--image-dir",
        type=Path,
        help="Directory containing all input images.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where metrics and figures will be written.",
    )
    parser.add_argument(
        "--week2-dir",
        type=Path,
        default=DEFAULT_WEEK2_DIR,
        help="Directory containing the completed Week 2 sfm_utils.py.",
    )
    parser.add_argument(
        "--week3-dir",
        type=Path,
        default=DEFAULT_WEEK3_DIR,
        help="Directory containing the completed Week 3 two_view_utils.py.",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=1600,
        help="Resize images so their long edge is at most this size. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=4000,
        help="Maximum number of SIFT features to retain per image.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.75,
        help="Lowe ratio-test threshold passed to the Week 2 matcher.",
    )
    parser.add_argument(
        "--focal-length-px",
        type=float,
        default=None,
        help="Optional focal length in pixels. Default is 1.2 times the image long edge.",
    )
    parser.add_argument(
        "--principal-point",
        nargs=2,
        type=float,
        metavar=("CX", "CY"),
        default=None,
        help="Optional principal point in pixels.",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=1.0,
        help="RANSAC threshold in pixels for essential matrix estimation.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.999,
        help="RANSAC confidence for essential matrix estimation.",
    )
    parser.add_argument(
        "--max-reprojection-error",
        type=float,
        default=4.0,
        help="Maximum reprojection error in pixels for keeping triangulated points.",
    )
    parser.add_argument(
        "--pnp-ransac-threshold",
        type=float,
        default=6.0,
        help="RANSAC reprojection threshold in pixels for third-view PnP.",
    )
    parser.add_argument(
        "--min-pnp-inliers",
        type=int,
        default=20,
        help="Minimum number of PnP inliers required to accept a new image.",
    )

    args = parser.parse_args()
    if args.max_image_size == 0:
        args.max_image_size = None
    if args.max_features < 1:
        parser.error("--max-features must be positive")
    if not 0.0 < args.ratio < 1.0:
        parser.error("--ratio must be between 0 and 1")
    if args.focal_length_px is not None and args.focal_length_px <= 0:
        parser.error("--focal-length-px must be positive")
    if args.ransac_threshold <= 0:
        parser.error("--ransac-threshold must be positive")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must be between 0 and 1")
    if args.max_reprojection_error <= 0:
        parser.error("--max-reprojection-error must be positive")
    if args.pnp_ransac_threshold <= 0:
        parser.error("--pnp-ransac-threshold must be positive")
    if args.min_pnp_inliers < 6:
        parser.error("--min-pnp-inliers must be at least 6")
    if args.image_dir is not None and not args.image_dir.is_dir():
        parser.error("--image-dir must be an existing directory")

    return args


def _median(values: np.ndarray) -> float | None:
    return float(np.median(values)) if len(values) else None


def _mean(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if len(values) else None


def expand_image_dir(image_dir: Path) -> list[Path]:
    supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in supported_ext and path.is_file()
        ]
    )
    if not images:
        raise ValueError(f"No supported image files found in directory: {image_dir}")
    return images


def compute_pairwise_match_graph(
    week2,
    week3,
    features: list,
    K: np.ndarray,
    ratio: float,
    ransac_threshold: float,
    confidence: float,
) -> list[PairwiseEdge]:
    edges: list[PairwiseEdge] = []
    n_images = len(features)
    for i in range(n_images - 1):
        for j in range(i + 1, n_images):
            matches = week2.match_descriptors(
                features[i].descriptors,
                features[j].descriptors,
                ratio=ratio,
            )
            if len(matches) < 8:
                continue

            pts_i, pts_j = week2.matched_keypoint_coords(
                features[i].keypoints,
                features[j].keypoints,
                matches,
            )
            E, inlier_mask = week3.estimate_essential_matrix(
                pts_i,
                pts_j,
                K,
                threshold=ransac_threshold,
                confidence=confidence,
            )
            inlier_count = int(np.sum(inlier_mask))
            edges.append(
                PairwiseEdge(
                    image_i=i,
                    image_j=j,
                    matches=matches,
                    inlier_mask=inlier_mask,
                    inlier_count=inlier_count,
                    essential_matrix=E,
                )
            )
    return edges


def choose_initial_pair(edges: list[PairwiseEdge]) -> PairwiseEdge:
    if not edges:
        raise ValueError("No valid pairwise edges found for an initial reconstruction.")
    return max(edges, key=lambda edge: edge.inlier_count)


def register_initial_pair(
    week2,
    week3,
    features,
    edge: PairwiseEdge,
    K: np.ndarray,
    max_reprojection_error: float,
) -> ReconstructionState:
    i, j = edge.image_i, edge.image_j
    pts_i, pts_j = week2.matched_keypoint_coords(
        features[i].keypoints,
        features[j].keypoints,
        edge.matches,
    )
    R, t, pose_mask = week3.recover_relative_pose(
        edge.essential_matrix,
        pts_i,
        pts_j,
        K,
        inlier_mask=edge.inlier_mask,
    )

    pts_i_pose = pts_i[pose_mask]
    pts_j_pose = pts_j[pose_mask]
    pose_matches = [match for match, keep in zip(edge.matches, pose_mask) if keep]

    points3d = week3.triangulate_points(pts_i_pose, pts_j_pose, K, R, t)
    errors_i = week3.compute_reprojection_errors(points3d, pts_i_pose, K, np.eye(3), np.zeros((3, 1)))
    errors_j = week3.compute_reprojection_errors(points3d, pts_j_pose, K, R, t)
    keep_mask = week3.filter_reconstructed_points(
        points3d,
        errors_i,
        errors_j,
        R,
        t,
        max_reprojection_error=max_reprojection_error,
    )

    kept_points = points3d[keep_mask]
    kept_colors = week3.sample_point_colours(features[i].image, pts_i_pose[keep_mask])

    tracks: dict[tuple[int, int], int] = {}
    next_point_id = 0
    for match, keep in zip(pose_matches, keep_mask):
        if not keep:
            continue
        tracks[(i, match.queryIdx)] = next_point_id
        tracks[(j, match.trainIdx)] = next_point_id
        next_point_id += 1

    return ReconstructionState(
        registered_images=[i, j],
        unregistered_images=[k for k in range(len(features)) if k not in {i, j}],
        camera_rotations={i: np.eye(3), j: R},
        camera_translations={i: np.zeros((3, 1)), j: t},
        tracks=tracks,
        points3d=kept_points,
        point_colors=kept_colors,
    )


def build_2d3d_correspondences_to_model(
    features,
    state: ReconstructionState,
    matches: list,
    registered_image: int,
    new_image: int,
) -> tuple[np.ndarray, np.ndarray]:
    points3d = []
    pts2d = []
    for match in matches:
        # match.queryIdx is from the first image in the pair,
        # match.trainIdx is from the second image.
        if (registered_image, match.queryIdx) in state.tracks:
            point_id = state.tracks[(registered_image, match.queryIdx)]
            points3d.append(state.points3d[point_id])
            pts2d.append(features[new_image].keypoints[match.trainIdx].pt)
        elif (registered_image, match.trainIdx) in state.tracks:
            point_id = state.tracks[(registered_image, match.trainIdx)]
            points3d.append(state.points3d[point_id])
            pts2d.append(features[new_image].keypoints[match.queryIdx].pt)

    if not points3d:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    return np.asarray(points3d, dtype=np.float64), np.asarray(pts2d, dtype=np.float64)


def choose_next_image(
    state: ReconstructionState,
    all_matches: dict[tuple[int, int], list],
) -> int | None:
    """
    triplet-based selection algorithm that selects the unregistered image with the 
    most 2D matches to already registered images, where the 2D matches must be to 
    keypoints that are part of existing 3D tracks in the model. This will ensure 
    that the next image has enough overlap with the existing reconstruction to be 
    successfully registered with PnP.
    """
    best_image = None
    best_support = 0

    # map each 3D point ID to the set of registered images that observe it
    point_observers = {}
    for (img_id, kp_idx), point_id in state.tracks.items():
        if img_id in state.registered_images:
            if point_id not in point_observers:
                point_observers[point_id] = set()
            point_observers[point_id].add(img_id)

    for image_id in state.unregistered_images:
        # To gather all unique 3D points that this unregistered image matches
        # to across all registered images, we can use a set.
        visible_points = set()
        for registered_id in state.registered_images:
            # safety net against missing pair in all_matches
            if (registered_id, image_id) in all_matches:
                pair = (registered_id, image_id)
            else:
                pair = (image_id, registered_id)

            matches = all_matches.get(pair)
            if matches is None:
                continue
            
            for match in matches:
                # Determine which side of the pair is the registered image
                if pair[0] == registered_id:
                    key = (registered_id, match.queryIdx)
                else:
                    key = (registered_id, match.trainIdx)

                if key in state.tracks:
                    point_id = state.tracks[key]
                    visible_points.add(point_id)

            if not visible_points:
                continue

            
            # To evaluate strongest local support, we can check how many of the 
            # 3D points visible to this unregistered image are also observed by each 
            # registered image. The registered image with the most shared points 
            # can be considered the strongest local support for this unregistered image.
            max_local_support = 0

            if len(state.registered_images) > 2:
                pair_count = {}
                for point_id in visible_points:
                    observers = list(point_observers.get(point_id, set()))
                    # Generate all unique pairs of registered images that observe this point
                    for obs1 in observers:
                        for obs2 in observers:
                            # to avoid duplicate pairs (obs1, obs2) and (obs2, obs1)
                            if obs1 < obs2:  
                                pair_count[(obs1, obs2)] = pair_count.get((obs1, obs2), 0) + 1
                if pair_count:
                    max_local_support = max(pair_count.values())

        # If there are no pairs of registered images that observe the same 3D points visible 
        # to this unregistered image,
        if max_local_support == 0:
            single_counts = {}
            for pt_id in visible_points:
                for obs in point_observers.get(pt_id, set()):
                    single_counts[obs] = single_counts.get(obs, 0) + 1
            if single_counts:
                max_local_support = max(single_counts.values())
        print(f'Image {image_id} max_local_support: {max_local_support}')
        # Update the best candidate
        if max_local_support > best_support:
            best_support = max_local_support
            best_image = image_id
            
   
    
    return best_image


def triangulate_and_append_new_points(
    week2,
    week3,
    features, 
    state: ReconstructionState,
    image_id: int,
    all_matches: dict[tuple[int, int], list],
    K: np.ndarray,
    max_reprojection_error: float,
) -> int:
    """
    Finds unused 2D matches between the newly registered image and all previously 
    registered images, triangulates them into 3D, and updates the global state.
    """
    # Steps:
    # 1. Isolate unmapped matches between the newly registered image and each previously registered image.
    # 2. Extract 2D coordinates of these matches and triangulate them into 3D points using the known camera poses.
    # 3. Triangulate into 3D
    # 4. Filter by reprojection error 
    # 5. Append and update state.tracks and state.points3d with the newly triangulated points that pass the 
    #    reprojection error threshold.

    R_new = state.camera_rotations[image_id]
    t_new = state.camera_translations[image_id]
    P_new = K @ np.hstack((R_new, t_new))

    new_points_count = 0
    for reg_id in state.registered_images:
        if reg_id == image_id:
            continue

        # Retrieve matches between the newly registered image and this registered image

        # safety net against missing pair in all_matches
        if (reg_id, image_id) in all_matches:
            pair = (reg_id, image_id)
        else:
            pair = (image_id, reg_id)

        matches = all_matches.get(pair)

        if matches is None:
            continue

        R_reg = state.camera_rotations[reg_id]
        t_reg = state.camera_translations[reg_id]
        P_reg = K @ np.hstack((R_reg, t_reg))

        # to isolate unmapped matches
        unmapped_matches = []
        unmapped_indices = [] # Stores tuple of (kp_reg, kp_new)

        for match in matches:
            if pair[0] == reg_id:
                kp_reg = match.queryIdx
                kp_new = match.trainIdx
            else:
                kp_reg = match.trainIdx
                kp_new = match.queryIdx

            # Check BOTH features to ensure we don't duplicate 3D geometry
            # by triangulating the same 3D point multiple times from different pairs of registered images.
            if (reg_id, kp_reg) in state.tracks or (image_id, kp_new) in state.tracks:
                continue

            unmapped_matches.append(match)
            unmapped_indices.append((kp_reg, kp_new))

        if not unmapped_matches:
            continue

        # extract 2D coordinates safely using the mapped indices
        pts_reg = np.array([features[reg_id].keypoints[reg_idx].pt for reg_idx, _ in unmapped_indices])
        pts_new = np.array([features[image_id].keypoints[new_idx].pt for _, new_idx in unmapped_indices])

        # triangulate them into 3D points using the known camera poses using absolute projection matrices P = K[R|t]
        points4d = cv2.triangulatePoints(P_reg, P_new, pts_reg.T, pts_new.T)
        points3d = (points4d[:3] / points4d[3]).T

        # to filter by reprojection error, we can compute the reprojection of these points into both views and check the error against the original 2D points.
        errors_reg = week3.compute_reprojection_errors(points3d, pts_reg, K, R_reg, t_reg)
        errors_new = week3.compute_reprojection_errors(points3d, pts_new, K, R_new, t_new)
        keep_mask = (errors_reg < max_reprojection_error) & (errors_new < max_reprojection_error)

        kept_points3d = points3d[keep_mask]
        if len(kept_points3d) == 0:
            continue

        # extract colours and add to global state
        kept_points_new = pts_new[keep_mask]
        kept_colours = week3.sample_point_colours(features[image_id].image, kept_points_new)

        start_index = len(state.points3d)
        state.points3d = np.vstack((state.points3d, kept_points3d))
        state.point_colors = np.vstack((state.point_colors, kept_colours))

        # Update state.tracks with the new points 
        # so that these features are considered "registered" in future iterations and won't 
        # be used for 2D-3D correspondences anymore.
        print(f'New points from single registered image {reg_id}: {int(np.sum(keep_mask))}')
        for (kp_reg, kp_new), keep in zip(unmapped_indices, keep_mask):
            if not keep:
                continue
                
            point_id = start_index
            state.tracks[(reg_id, kp_reg)] = point_id
            state.tracks[(image_id, kp_new)] = point_id

            start_index += 1
            new_points_count += 1
    print(f'Total new points added {new_points_count}')
    return new_points_count


def register_next_image(
    week2,
    week3,
    features,
    state: ReconstructionState,
    image_id: int,
    all_matches: dict[tuple[int, int], list],
    K: np.ndarray,
    pnp_ransac_threshold: float,
    confidence: float,
    min_pnp_inliers: int,
    ) -> bool:
    correspondences3d = []
    correspondences2d = []
    for reg_id in state.registered_images:
        pair = (reg_id, image_id) if (reg_id, image_id) in all_matches else (image_id, reg_id)
        matches = all_matches.get(pair)
        if matches is None:
            continue
        pts3d, pts2d = build_2d3d_correspondences_to_model(
            features,
            state,
            matches,
            registered_image=reg_id,
            new_image=image_id,
        )
        print(f'3D points between registered {reg_id} and unregistered image {image_id}: {len(pts3d)}')
        if len(pts3d):
            correspondences3d.append(pts3d)
            correspondences2d.append(pts2d)

    if not correspondences3d:
        return False

    points3d = np.vstack(correspondences3d)
    pts2d = np.vstack(correspondences2d)
    if len(points3d) < 6:
        return False

    R_new, t_new, pnp_mask = week3.estimate_camera_pose_pnp(
        points3d,
        pts2d,
        K,
        threshold=pnp_ransac_threshold,
        confidence=confidence,
    )

    inlier_count = int(np.sum(pnp_mask))
    print(f'Inlier count:{inlier_count}')
    if inlier_count < min_pnp_inliers:
        return False

    state.camera_rotations[image_id] = R_new
    state.camera_translations[image_id] = t_new
    state.registered_images.append(image_id)
    state.unregistered_images.remove(image_id)

    triangulate_and_append_new_points(
        week2,
        week3,
        features, 
        state, 
        image_id, 
        all_matches, 
        K, 
        max_reprojection_error=4.0 # Or pass this down from args
    )
    
    return True


def incremental_reconstruction(args: argparse.Namespace) -> ReconstructionState:
    week2 = load_week2_module(args.week2_dir)
    week3 = load_week3_module(args.week3_dir)
    output_dir = week2.ensure_dir(args.output_dir)

    images = args.images
    if args.image_dir is not None:
        images = expand_image_dir(args.image_dir)
    elif images is not None and len(images) == 1 and images[0].is_dir():
        images = expand_image_dir(images[0])

    features = week2.precompute_image_features(
        images,
        max_features=args.max_features,
        max_image_size=args.max_image_size,
    )

    K = week3.make_camera_matrix(
        features[0].image.shape,
        focal_length_px=args.focal_length_px,
        principal_point=None if args.principal_point is None else tuple(args.principal_point),
    )

    edges = compute_pairwise_match_graph(
        week2,
        week3,
        features,
        K,
        args.ratio,
        args.ransac_threshold,
        args.confidence,
    )

    pairwise_matches: dict[tuple[int, int], list] = {}
    for edge in edges:
        pairwise_matches[(edge.image_i, edge.image_j)] = edge.matches

    initial_edge = choose_initial_pair(edges)
    state = register_initial_pair(
        week2,
        week3,
        features,
        initial_edge,
        K,
        max_reprojection_error=args.max_reprojection_error,
    )

    plotter = ReconstructionPlotter(window_name="Live GF4 Reconstruction")
    plotter.update(state, K)
    time.sleep(2)
    while state.unregistered_images:
        #next_image = choose_next_image(features, state, pairwise_matches)
        next_image = choose_next_image(state, pairwise_matches)
        if next_image is None:
            print('Next image not found.')
            break
        accepted = register_next_image(
            week2,
            week3,
            features,
            state,
            next_image,
            pairwise_matches,
            K,
            args.pnp_ransac_threshold,
            args.confidence,
            args.min_pnp_inliers,
        )
        if not accepted:
            break
        print(f"Registered image {next_image} ({len(state.registered_images)} total), {len(state.points3d)} points")
        # to consider appending metrics to a CSV after each successful registration
        plotter.update(state, K)
        time.sleep(2)
    print("Reconstruction complete. Close the 3D viewer window to continue.")
    plotter.vis.run() 
    o3d.io.write_point_cloud(output_dir/"reconstruction.ply", plotter.point_cloud)
    plotter.close()

    return state


def main() -> int:
    args = parse_args()
    try:
        state = incremental_reconstruction(args)
    except NotImplementedError as exc:
        print(f"\nStarter-code TODO reached: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Incremental reconstruction scaffold complete")
    print(f"  registered images: {len(state.registered_images)}")
    print(f"  remaining images: {len(state.unregistered_images)}")
    print(f"  reconstructed points: {len(state.points3d)}")
    print(f"  wrote: {args.output_dir}")


    # camera_poses = []
    # for img_id in state.registered_images:
    #     camera_poses.append((
    #         f"Image {img_id}", 
    #         state.camera_rotations[img_id], 
    #         state.camera_translations[img_id]
    #     ))

    # # Save outputs for your report
    # write_ply(args.output_dir / "final_reconstruction.ply", state.points3d, state.point_colors)
    # plot_multi_view_reconstruction(state.points3d, state.point_colors, camera_poses, args.output_dir / "camera_trajectory.png")


    return 0

        

if __name__ == "__main__":
    raise SystemExit(main())


