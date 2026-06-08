from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
import sys
import cv2

import numpy as np

import time
import matplotlib.pyplot as plt

import ba_utils as ba
import re_utils as re
import csv

import os
os.environ["XDG_SESSION_TYPE"] = "x11"


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
            frustum = self._build_camera_frustum(R, t, scale=1)
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
    parser.add_argument(
        "--draw-graph",
        action="store_true",
        help="Draw and save the pairwise match graph after matching.",
    )
    parser.add_argument(
        "--graph-output",
        type=Path,
        default=None,
        help="Path to save the pairwise match graph image.",
    )
    parser.add_argument(
        "--bundle-adjustment",
        action="store_true",
        default=False,
        help= 'Conduct Bundle Adjustment'
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

def draw_pairwise_match_graph(week2, edges: list[PairwiseEdge], output_path: Path | None = None) -> None:
    edges = [e for e in edges if e.inlier_count > 750]
    if not edges:
        return

    nodes = set()
    for e in edges:
        nodes.add(e.image_i)
        nodes.add(e.image_j)
    nodes = sorted(nodes)
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {node: (np.cos(a), np.sin(a)) for node, a in zip(nodes, angles)}

    weights = [e.inlier_count for e in edges]
    norm = plt.Normalize(vmin=min(weights), vmax=max(weights))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.axis("off")

    xs = [pos[node][0] for node in nodes]
    ys = [pos[node][1] for node in nodes]
    ax.scatter(xs, ys, s=300, c="tab:blue")

    for node in nodes:
        x, y = pos[node]
        ax.text(x * 1.12, y * 1.12, str(node), ha="center", va="center")

    for e in edges:
        i = e.image_i
        j = e.image_j
        x1, y1 = pos[i]
        x2, y2 = pos[j]
        w = e.inlier_count
        color = cmap(norm(w))
        lw = 0.5 + 4.0 * (w / max(weights)) if max(weights) > 0 else 1.0
        ax.plot([x1, x2], [y1, y2], c=color, linewidth=lw, alpha=0.85)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(weights)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("RANSAC inlier count")

    if output_path is not None:
        week2.ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

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
    pair: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points3d = []
    pts2d = []
    point_ids = []
    new_kp_ids = []
    for match in matches:
        # match.queryIdx is from the first image in the pair,
        # match.trainIdx is from the second image.
        if pair[0] == registered_image:
            kp_reg = match.queryIdx
            kp_new = match.trainIdx
        else:
            kp_reg = match.trainIdx
            kp_new = match.queryIdx
        
        track_key = (registered_image, kp_reg)
        if track_key not in state.tracks:
            continue
        
        point_id = state.tracks[track_key]
        points3d.append(state.points3d[point_id])
        pts2d.append(features[new_image].keypoints[kp_new].pt)
        point_ids.append(point_id)
        new_kp_ids.append(kp_new)

    if not points3d:
        return (np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64), 
                np.empty((0,), dtype=np.float64),np.empty((0,), dtype=np.float64))
    return (np.asarray(points3d, dtype=np.float64), np.asarray(pts2d, dtype=np.float64), 
            np.asarray(point_ids, dtype=np.float64), np.asarray(new_kp_ids, dtype=np.float64))


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

    Indicators used:
     - shared RANSAC inliers
     - 
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

def observations_by_image(state):
    obs = {}
    for (img_id, kp_idx), point_id in state.tracks.items():
        obs.setdefault(img_id, set()).add(point_id)
    return obs

def choose_active_camera_ids(state, new_image_id, max_neighbors=4):
    '''
    Select the local camera window for bundle adjustment.
    
    Returns:
        [new_image_id, neighbour_1, neightbour_2, ..., neighbour_max]
    '''
    obs = observations_by_image(state) 
    # For now rebuild every time. Could upgrade to a maintained state in ReconstructionState. 
    # state.image_to_points and state.point_to_images
    new_points = obs.get(new_image_id, set())

    scores = []
    for cam_id in state.registered_images:
        if cam_id == new_image_id:
            continue
        # score candidates based number of shared visible reconstructed 3d points.
        shared = len(new_points & obs.get(cam_id, set()))
        if shared > 0:
            scores.append((shared, cam_id))

    scores.sort(reverse=True)
    neighbors = [cam_id for _, cam_id in scores[:max_neighbors]]
    return [new_image_id] + neighbors

def choose_active_point_ids(state, active_camera_ids, min_active_obs=2):
    '''
    Select the 3D points belonging to the BA window.
    
    Returns:
        [(img_idx, kp_idx), ... ]
    '''
    
    active_set = set(active_camera_ids)

    point_to_active_obs = {}
    for (img_id, kp_idx), point_id in state.tracks.items():
        if img_id not in active_set:
            continue
        point_to_active_obs.setdefault(point_id, []).append((img_id, kp_idx))

    active_point_ids = [
        point_id
        for point_id, obs in point_to_active_obs.items()
        if len(obs) >= min_active_obs
    ]
    return active_point_ids, point_to_active_obs

def bundle_adjustment(
    state: ReconstructionState,
    features: list,
    K: np.array,
    new_image_id: int,
    window_size: int = 5
) -> None:
    """
    Perform Local Bundle Adjustment
    """
    # Find (window_size - 1) Nearest neighbour cameras based on shared point appearances
    active_camera_ids = choose_active_camera_ids(state, new_image_id, window_size - 1)
    # Find all points that appear in this active set of cameras
    active_point_ids, point_to_active_obs = choose_active_point_ids(state, active_camera_ids)
    
    if len(active_camera_ids) < 2 or len(active_point_ids) < 2:
        return
    
    # create local dictionaries
    cam_id_to_local = {cam_id: i for i, cam_id in enumerate(active_camera_ids)}
    pt_id_to_local = {pt_id: i for i, pt_id in enumerate(active_point_ids)}
    
    # perpare active camera params for packing --> BA
    camera_params0 = []
    for cam_id in active_camera_ids:
        R = state.camera_rotations[cam_id]
        t = state.camera_translations[cam_id].reshape(3)
        rvec, _ = cv2.Rodrigues(R)
        camera_params0.append(np.hstack([rvec.ravel(), t]))

    camera_params0 = np.asarray(camera_params0, dtype=np.float64)
    points3d0 = state.points3d[active_point_ids].astype(np.float64)
    
    # For each local observation we need:
    #   - which local camera sees it
    #   - which local 3D point it is
    #   - what the measured 2D pixel location is
    
    camera_indices = []
    point_indices = []
    points_2d = []
    
    for point_id in active_point_ids:
        local_pt_idx = pt_id_to_local[point_id]
        for img_id, kp_idx in point_to_active_obs[point_id]:
            if img_id not in cam_id_to_local:
                continue
            local_cam_idx = cam_id_to_local[img_id]
            uv = features[img_id].keypoints[kp_idx].pt

            camera_indices.append(local_cam_idx)
            point_indices.append(local_pt_idx)
            points_2d.append(uv)

    camera_indices = np.asarray(camera_indices, dtype=np.int32)
    point_indices = np.asarray(point_indices, dtype=np.int32)
    points_2d = np.asarray(points_2d, dtype=np.float64)
    
    # These are index arrays used by the least squares and the 2d observed points they correspond to
    
    if len(points_2d) == 0:
        return
    # Fix a camera to stabilize the optimization
    
    fixed_camera_mask = np.zeros(len(active_camera_ids), dtype=bool)
    fixed_camera_mask[1] = True # Fix the strongest neighbour (should be the most refined?) 
    fixed_camera_params = camera_params0.copy()
    
    n_cams = camera_params0.shape[0]
    n_pts = points3d0.shape[0]
    
    result = ba.least_squares_fit(camera_params0,
                                  points3d0,
                                  n_cams,
                                  n_pts,
                                  camera_indices,
                                  point_indices,
                                  points_2d,
                                  K,
                                  fixed_camera_mask,
                                  fixed_camera_params)
    
    cam_params_opt, pts3d_opt = ba.unpack_params(result.x, n_cams, n_pts)
    
    cam_params_opt[fixed_camera_mask] = fixed_camera_params[fixed_camera_mask]
    
    #Write optimized camera parameters in global reconstruction
    for cam_id, cam_param in zip(active_camera_ids, cam_params_opt):
        rvec = cam_param[:3]
        tvec = cam_param[3:].reshape(3, 1)
        R = ba.rodrigues_to_R(rvec)

        state.camera_rotations[cam_id] = R
        state.camera_translations[cam_id] = tvec

    #Write optimized 3D point coordinates
    for pt_id, X in zip(active_point_ids, pts3d_opt):
        state.points3d[pt_id] = X
    
    return

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

    C_new = (-R_new.T @ t_new).ravel()
    
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

        # optical center of registered camera
        C_reg = (-R_reg.T @ t_reg).ravel()


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
        #print(points3d.shape)

        # to filter by reprojection error, we can compute the reprojection of these points into both views and check the error against the original 2D points.
        errors_reg = week3.compute_reprojection_errors(points3d, pts_reg, K, R_reg, t_reg)
        errors_new = week3.compute_reprojection_errors(points3d, pts_new, K, R_new, t_new)
        reproj_mask = (errors_reg < max_reprojection_error) & (errors_new < max_reprojection_error)
        
        finite_mask = np.isfinite(points3d).all(axis=1)
        _, z_reg = week3.compute_depths(points3d, R_reg, t_reg)
        _, z_new = week3.compute_depths(points3d, R_new, t_new)
        depth_mask = (z_reg > 0) & (z_new > 0)

        # triangulation angle check
        v_reg = points3d - C_reg
        v_new = points3d - C_new

        dot_prod = np.sum(v_reg * v_new, axis=1)
        norm_reg = np.linalg.norm(v_reg, axis=1)
        norm_new = np.linalg.norm(v_new, axis=1)

        cos_theta = dot_prod / (norm_reg * norm_new)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.rad2deg(np.arccos(cos_theta))

        angle_mask = theta > 2.0
        
        keep_mask = reproj_mask & finite_mask & depth_mask & angle_mask
        
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
    #correspondences3d = []
    #correspondences2d = []
    correspondence_rows = []
    for reg_id in state.registered_images:
        pair = (reg_id, image_id) if (reg_id, image_id) in all_matches else (image_id, reg_id)
        matches = all_matches.get(pair)
        if matches is None:
            continue
        pts3d, pts2d, point_ids, kp_ids = build_2d3d_correspondences_to_model(features,
                                                                              state,
                                                                              matches,
                                                                              registered_image=reg_id,
                                                                              new_image=image_id,
                                                                              pair= pair
                                                                              )  
                                                                              
        print(f'3D points between registered {reg_id} and unregistered image {image_id}: {len(pts3d)}')
        #if len(pts3d):
        #    correspondences3d.append(pts3d)
        #    correspondences2d.append(pts2d)
        for X, x, pid, kid in zip(pts3d, pts2d, point_ids, kp_ids):
            correspondence_rows.append((pid, kid, X, x))


    #if not correspondences3d:
    #    return False

    #points3d = np.vstack(correspondences3d)
    #pts2d = np.vstack(correspondences2d)
    best_by_pid = {}
    for pid, kid, X, x in correspondence_rows:
        if pid not in best_by_pid:
            best_by_pid[pid] = (kid, X, x)

    rows = list(best_by_pid.items())

    seen_kp = set()
    final_X, final_x = [], []
    for _, (kid, X, x) in rows:
        if kid in seen_kp:
            continue
        seen_kp.add(kid)
        final_X.append(X)
        final_x.append(x)

    points3d = np.asarray(final_X, dtype=np.float64)
    pts2d = np.asarray(final_x, dtype=np.float64)
    
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

    inlier_X = points3d[pnp_mask.ravel().astype(bool)]
    inlier_x = pts2d[pnp_mask.ravel().astype(bool)]
    errs = week3.compute_reprojection_errors(inlier_X, inlier_x, K, R_new, t_new)

    if np.median(errs) > 3.0 or np.mean(errs) > 5.0:
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

def rank_candidate_images(state, pairwise_matches):
    scores = []
    for image_id in state.unregistered_images:
        score = 0
        for reg_id in state.registered_images:
            pair = (reg_id, image_id) if (reg_id, image_id) in pairwise_matches else (image_id, reg_id)
            matches = pairwise_matches.get(pair)
            
            if matches is not None:
                score += len(matches)
                
        if score > 0:
            scores.append((score, image_id))
            
    scores.sort(reverse=True)
    
    return [image_id for _, image in scores]

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

    if args.draw_graph:
        out = (Path(args.output_dir) / "pairwise_match_graph.png")
        draw_pairwise_match_graph(week2, edges, out)

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
    # time.sleep(2)
    plotter.vis.poll_events()
    plotter.vis.update_renderer()

    # initialise csv
    csv_path = args.output_dir / "incremental_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "registered_image_count",
            "newest_image_id",
            "total_3d_points",
            "global_mean_px",
            "global_median_px",
            "point_mean_px",
            "point_median_px"
        ])


    while state.unregistered_images:
        #next_image = choose_next_image(features, state, pairwise_matches)
        #next_image = choose_next_image(state, pairwise_matches)
        
        next_candidates = rank_candidate_images(state, pairwise_matches)
        accepted_any = False
        for next_image in next_candidates:
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
            if accepted:
                accepted_any = True
                break
        
        if not accepted_any:
            break
        
        if len(state.registered_images) > 3 and len(state.registered_images) % 4 == 0 and args.bundle_adjustment:
            bundle_adjustment(state, features, K, next_image, window_size=3)
        print(f"Registered image {next_image} ({len(state.registered_images)} total), {len(state.points3d)} points")
        # TODO: consider appending metrics to a CSV after each successful registration
        g_mean, g_med = re.compute_global_reprojection_error(state, features, K, week3)
        p_mean, p_med = re.compute_pointwise_reprojection_error(state, features, K, week3)
        print(f"  -> Global Error:   Mean {g_mean:.3f}px | Median {g_med:.3f}px")
        print(f"  -> Point-wise Err: Mean {p_mean:.3f}px | Median {p_med:.3f}px")

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                len(state.registered_images),
                next_image,
                len(state.points3d),
                f"{g_mean:.4f}",
                f"{g_med:.4f}",
                f"{p_mean:.4f}",
                f"{p_med:.4f}"
            ])

        plotter.update(state, K)
        # time.sleep(2)
        plotter.vis.poll_events()
        plotter.vis.update_renderer()

    print("Reconstruction complete. Close the 3D viewer window to continue.")
    plotter.vis.run() 
    o3d.io.write_point_cloud(output_dir/"reconstruction.ply", plotter.point_cloud)
    plotter.close()

    return state


def main() -> int:
    args = parse_args()
    week2 = load_week2_module(args.week2_dir)
    week3 = load_week3_module(args.week3_dir)
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

    g_mean, g_med = re.compute_global_reprojection_error(state, features, K, week3)
    p_mean, p_med = re.compute_pointwise_reprojection_error(state, features, K, week3)

    print("Final Reprojection Errors:")
    print(f"  Global   - Mean: {g_mean:.3f}px, Median: {g_med:.3f}px")
    print(f"  Point-wise - Mean: {p_mean:.3f}px, Median: {p_med:.3f}px")


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

    # TODO: 
    # 1. no of input images
    # 2. no of registered images
    # 3. no of sparse points
    # 4. csv of shared ransac inliers between triplets
    #     - every line is a registered image

    point_observers = {}

    for (img_id, kp_idx), point_id in state.tracks.items():
        if img_id in state.registered_images:
            point_observers.setdefault(point_id, []).append(img_id)

    csv_rows = []
    for img_id in state.registered_images:
        total_points = 0
        triplet_shared_points = 0

        for (trk_img, kp_idx), point_id in state.tracks.items():
            if trk_img == img_id:
                total_points += 1
                if point_id in point_observers:
                    triplet_shared_points += 1
                
        csv_rows.append({
            "image_id": img_id, 
            "total_3d_points_observed": total_points, 
            "triplet_shared_points": triplet_shared_points
            })

    csv_path = args.output_dir / "triplet_inliers.csv"
    week2.save_csv(csv_path, csv_rows)
    print(f"Saved triplet inliers CSV to: {csv_path}")

    

    return 0

        

if __name__ == "__main__":
    raise SystemExit(main())


