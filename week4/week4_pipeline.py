from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

from week3.two_view_utils import (
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
)

DEFAULT_WEEK2_DIR = Path(__file__).resolve().parents[1] / "week2"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GF4 Week 4 incremental sparse reconstruction pipeline."
    )
    parser.add_argument(
        "--images",
        nargs="+",
        type=Path,
        required=True,
        help="Paths to all images in the sequence.",
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

    return args


def _median(values: np.ndarray) -> float | None:
    return float(np.median(values)) if len(values) else None


def _mean(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if len(values) else None


def compute_pairwise_match_graph(
    week2,
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
            E, inlier_mask = estimate_essential_matrix(
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
    R, t, pose_mask = recover_relative_pose(
        edge.essential_matrix,
        pts_i,
        pts_j,
        K,
        inlier_mask=edge.inlier_mask,
    )

    pts_i_pose = pts_i[pose_mask]
    pts_j_pose = pts_j[pose_mask]
    pose_matches = [match for match, keep in zip(edge.matches, pose_mask) if keep]

    points3d = triangulate_points(pts_i_pose, pts_j_pose, K, R, t)
    errors_i = compute_reprojection_errors(points3d, pts_i_pose, K, np.eye(3), np.zeros((3, 1)))
    errors_j = compute_reprojection_errors(points3d, pts_j_pose, K, R, t)
    keep_mask = filter_reconstructed_points(
        points3d,
        errors_i,
        errors_j,
        R,
        t,
        max_reprojection_error=max_reprojection_error,
    )

    kept_points = points3d[keep_mask]
    kept_colors = sample_point_colours(features[i].image, pts_i_pose[keep_mask])

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
    image_i: int,
    image_j: int,
) -> tuple[np.ndarray, np.ndarray]:
    points3d = []
    pts2d = []
    for match in matches:
        key = (image_i, match.queryIdx)
        if key not in state.tracks:
            continue
        point_id = state.tracks[key]
        points3d.append(state.points3d[point_id])
        pts2d.append(features[image_j].keypoints[match.trainIdx].pt)

    if not points3d:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    return np.asarray(points3d, dtype=np.float64), np.asarray(pts2d, dtype=np.float64)


def choose_next_image(
    state: ReconstructionState,
    all_matches: dict[tuple[int, int], list],
) -> int | None:
    best_image = None
    best_support = 0
    for image_id in state.unregistered_images: # search all images in un-registered images
        support = 0
        for registered_id in state.registered_images:  
            pair = (registered_id, image_id) if (registered_id, image_id) in all_matches else (image_id, registered_id)
            matches = all_matches.get(pair) # find matches in this pair
            if matches is None:
                continue
            support += sum(
                1
                for match in matches
                if (registered_id, match.queryIdx if pair[0] == registered_id else match.trainIdx) in state.tracks
            )
        if support > best_support:
            best_support = support
            best_image = image_id
    
    return best_image


def triangulate_and_append_new_points(
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
    R_new = state.camera_rotations[image_id]
    t_new = state.camera_translations[image_id]
    P_new = K @ np.hstack((R_new, t_new))

    new_points_count = 0
    for reg_id in state.registered_images:
        if reg_id == image_id:
            continue

        # Retrieve matches between the newly registered image and this registered image
        pair = (reg_id, image_id) if (reg_id, image_id) in all_matches else (image_id, reg_id)
        matches = all_matches.get(pair)
        if matches is None:
            continue

def register_next_image(
    week2,
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
        if pair[0] == reg_id:
            pts3d, pts2d = build_2d3d_correspondences_to_model(features, state, matches, reg_id, image_id)
        else:
            pts3d, pts2d = build_2d3d_correspondences_to_model(features, state, matches, image_id, reg_id)
            # swap because build_2d3d_correspondences_to_model assumes queryIdx is registered image
            pts3d, pts2d = pts3d, pts2d
        if len(pts3d):
            correspondences3d.append(pts3d)
            correspondences2d.append(pts2d)

    if not correspondences3d:
        return False

    points3d = np.vstack(correspondences3d)
    pts2d = np.vstack(correspondences2d)
    if len(points3d) < 6:
        return False

    R_new, t_new, pnp_mask = estimate_camera_pose_pnp(
        points3d,
        pts2d,
        K,
        threshold=pnp_ransac_threshold,
        confidence=confidence,
    )

    inlier_count = int(np.sum(pnp_mask))
    if inlier_count < min_pnp_inliers:
        return False

    state.camera_rotations[image_id] = R_new
    state.camera_translations[image_id] = t_new
    state.registered_images.append(image_id)
    state.unregistered_images.remove(image_id)

    # TODO: triangulate new points between image_id and registered views,
    # update state.tracks and state.points3d accordingly.
    
    return True


def incremental_reconstruction(args: argparse.Namespace) -> ReconstructionState:
    output_dir = ensure_dir(args.output_dir)
    week2 = load_week2_module(args.week2_dir)

    features = week2.precompute_image_features(
        args.images,
        max_features=args.max_features,
        max_image_size=args.max_image_size,
    )

    K = make_camera_matrix(
        features[0].image.shape,
        focal_length_px=args.focal_length_px,
        principal_point=None if args.principal_point is None else tuple(args.principal_point),
    )

    edges = compute_pairwise_match_graph(
        week2,
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
        features,
        initial_edge,
        K,
        max_reprojection_error=args.max_reprojection_error,
    )

    while state.unregistered_images:
        next_image = choose_next_image(features, state, pairwise_matches)
        if next_image is None:
            break
        accepted = register_next_image(
            week2,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


