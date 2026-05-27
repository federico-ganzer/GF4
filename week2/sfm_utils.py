"""Utility functions for GF4 Week 2 pairwise SfM front-end.

This file is intentionally a starter scaffold. Basic file handling and a few
plotting helpers are provided. The core SfM-front-end steps are marked with
TODO and should be completed by students.
"""
# init
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class ImageFeatures:
    """Image, keypoints, and descriptors for one input image."""

    path: Path
    image: np.ndarray
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray


@dataclass
class PairAnalysis:
    """Container for pairwise matching and epipolar-geometry results."""

    image_i: str
    image_j: str
    keypoints_i: int
    keypoints_j: int
    raw_matches: int
    filtered_matches: int
    ransac_inliers: int
    inlier_ratio: float
    mean_epipolar_error_all: float | None
    median_epipolar_error_all: float | None
    mean_epipolar_error_inliers: float | None
    median_epipolar_error_inliers: float | None
    max_epipolar_error_inliers: float | None
    fundamental_matrix: list[list[float]] | None

    def as_dict(self) -> dict:
        return {
            "image_i": self.image_i,
            "image_j": self.image_j,
            "keypoints_i": self.keypoints_i,
            "keypoints_j": self.keypoints_j,
            "raw_matches": self.raw_matches,
            "filtered_matches": self.filtered_matches,
            "ransac_inliers": self.ransac_inliers,
            "inlier_ratio": self.inlier_ratio,
            "mean_epipolar_error_all": self.mean_epipolar_error_all,
            "median_epipolar_error_all": self.median_epipolar_error_all,
            "mean_epipolar_error_inliers": self.mean_epipolar_error_inliers,
            "median_epipolar_error_inliers": self.median_epipolar_error_inliers,
            "max_epipolar_error_inliers": self.max_epipolar_error_inliers,
            "fundamental_matrix": self.fundamental_matrix,
        }

    def csv_dict(self) -> dict:
        """Return scalar fields suitable for CSV output."""
        data = self.as_dict()
        data["fundamental_matrix"] = (
            "" if self.fundamental_matrix is None else str(self.fundamental_matrix)
        )
        return data


def ensure_dir(path: Path) -> Path:
    """Create an output directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_image_paths(image_dir: Path, max_images: int | None = None) -> list[Path]:
    """Return sorted image paths from a directory."""
    image_dir = Path(image_dir)
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    paths = sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if max_images is not None:
        paths = paths[:max_images]
    if not paths:
        raise ValueError(f"No images found in {image_dir}")
    return paths


def load_image(path: Path, max_size: int | None = None) -> np.ndarray:
    """Load an image with OpenCV in BGR order, optionally resizing the long edge."""
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if max_size is not None:
        height, width = image.shape[:2]
        scale = max_size / max(height, width)
        if scale < 1.0:
            image = cv2.resize(
                image,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            
    return image


def save_csv(path: Path, rows: Iterable[dict]) -> None:
    """Save a list of dictionaries as CSV."""
    rows = list(rows)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def detect_sift_features(
    image: np.ndarray,
    max_features: int = 4000,
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """Detect SIFT keypoints and descriptors.

    TODO: Complete this function. DONE

    Hints:
    - Convert the image to grayscale.
    - Create a SIFT detector with cv2.SIFT_create(nfeatures=max_features).
    - Return keypoints and descriptors from detector.detectAndCompute(...).
    - If no descriptors are found, return an empty array with shape (0, 128).
    - If OpenCV returns slightly more than max_features, keep only the first
      max_features keypoints and matching descriptor rows.
    """
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create(max_features)
    kps, des = sift.detectAndCompute(gray, mask=None)
    
    if des is None or len(des) ==0:
        return kps, np.empty((0, 128))
    
    if len(kps) > max_features:
        kps = kps[:max_features]
        des = des[:max_features]
    
    return kps, des
    

def precompute_image_features(
    image_paths: list[Path],
    max_features: int = 4000,
    max_image_size: int | None = 1600,
) -> list[ImageFeatures]:
    """Load each image and compute SIFT features once.

    TODO: Complete this function after implementing detect_sift_features. DONE

    Dataset mode should use this function so SIFT is not recomputed for the
    same image in every pair.
    """
    features = []
    
    for image_path in image_paths:
        image = load_image(image_path, max_image_size)
    
        kps, des = detect_sift_features(image= image, max_features=max_features)
        
        features.append(ImageFeatures(path= image_path, image= image, keypoints= kps, descriptors= des))
    return features


def raw_descriptor_matches(desc1: np.ndarray, desc2: np.ndarray) -> list[cv2.DMatch]:
    """Return one nearest-neighbour match per descriptor before Lowe filtering.

    TODO: Complete this function. DONE

    Hints:
    - Handle empty descriptor arrays by returning an empty list.
    - For SIFT descriptors, use cv2.BFMatcher(cv2.NORM_L2).
    - Use matcher.match(desc1, desc2) to get the best match in image 2 for
      each descriptor in image 1.
    - Return the matches sorted by descriptor distance.
    """

    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) < 2:
        return []
    
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key= lambda d : d.distance)
    
    return matches
    

def match_descriptors(
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio: float = 0.75,
) -> list[cv2.DMatch]:
    """Match SIFT descriptors using Lowe's ratio test.

    TODO: Complete this function. DONE

    Hints:
    - Handle empty descriptor arrays by returning an empty list.
    - For SIFT descriptors, use cv2.BFMatcher(cv2.NORM_L2).
    - Use knnMatch(desc1, desc2, k=2) for the ratio test.
    - Keep a match when best_distance < ratio * second_best_distance.
    """
    # added to test for < 2. Lowe's test doesn't work for < 2
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) < 2:
        return []
    
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(desc1, desc2, k=2) # returns list of 2 best matching train descriptors (desc2) for each query descriptor (desc1)
    good_matches = []
    for d1, d2 in matches:
        if d1.distance < ratio * d2.distance:
            good_matches.append(d1)            
    
    return good_matches


def count_raw_matches(desc1: np.ndarray, desc2: np.ndarray) -> int:
    """Return the number of descriptors that can be matched before filtering.

    TODO: Complete this function. DONE

    A simple definition is len(raw_descriptor_matches(desc1, desc2)). This
    gives a useful denominator for comparing raw and filtered matching.
    """
    
    return len(raw_descriptor_matches(desc1, desc2))


def matched_keypoint_coords(
    keypoints1: list[cv2.KeyPoint],
    keypoints2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert OpenCV matches into aligned Nx2 coordinate arrays.

    TODO: Complete this function. DONE

    Remember: cv2.KeyPoint.pt is (x, y), not (row, column).
    """
    
    pts1 = np.array([keypoints1[m.queryIdx].pt for m in matches], dtype= np.float32)
    pts2 = np.array([keypoints2[m.trainIdx].pt for m in matches], dtype= np.float32)
    
    return pts1, pts2


def estimate_fundamental_ransac(
    pts1: np.ndarray,
    pts2: np.ndarray,
    threshold: float = 1.0,
    confidence: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the fundamental matrix with OpenCV RANSAC.

    TODO: Complete this function. DONE

    Return:
    - F: 3x3 fundamental matrix
    - inlier_mask: boolean array of shape (N,)
    """
    
    F, mask = cv2.findFundamentalMat(
        pts1, 
        pts2, 
        method= cv2.FM_RANSAC, 
        ransacReprojThreshold= threshold, 
        confidence= confidence,
        maxIters= 1000
        )
    
    return F, mask


def compute_epipolar_errors(
    F: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> np.ndarray:
    """Compute point-to-epipolar-line distances in image 2.

    TODO: Complete this function. DONE

    For each point x1 in image 1, compute the epipolar line l2 = F x1.
    Then compute the distance from the corresponding x2 to l2.
    """
    
    
    # Computes epipolar lines in image 2
    l2s = F @ np.hstack((pts1, np.ones((len(pts1), 1)))).T
    # Calculate point-to-line distance
    err = np.abs(np.sum(l2s.T * np.hstack((pts2, np.ones((len(pts2), 1)))), axis=1)) / np.linalg.norm(l2s[:2], axis=0)
    
    return err
    


def draw_keypoints(
    image: np.ndarray,
    keypoints: list[cv2.KeyPoint],
    output_path: Path,
) -> None:
    """Save a keypoint visualisation."""
    ensure_dir(output_path.parent)
    vis = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    cv2.imwrite(str(output_path), vis)


def draw_matches(
    image1: np.ndarray,
    keypoints1: list[cv2.KeyPoint],
    image2: np.ndarray,
    keypoints2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    output_path: Path,
    max_draw: int = 80,
) -> None:
    """Save a feature-match visualisation."""
    ensure_dir(output_path.parent)
    matches_to_draw = sorted(matches, key=lambda m: m.distance)[:max_draw]
    vis = cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(output_path), vis)


def draw_epipolar_lines(
    image1: np.ndarray,
    image2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    F: np.ndarray,
    output_path: Path,
    max_lines: int = 20,
) -> None:
    """Save an epipolar-line visualisation.

    TODO: Complete this function.

    Hints:
    - Sample up to max_lines corresponding points.
    - For each x1, draw l2 = F x1 in image 2.
    - Draw the corresponding x2 point on image 2.
    - A simple Matplotlib figure with image1 and image2 side by side is enough.
    """
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)
    # Make images coloured
    img1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # Sample up to max_lines
    if len(pts1) > max_lines:
        idx = np.random.choice(len(pts1), max_lines, replace=False)
    else:
        idx = np.arange(len(pts1))
    
    pts1_sample = pts1[idx]
    pts2_sample = pts2[idx]

    # Compute l2 in image 2 for the sampled points
    pts1_homog = np.hstack((pts1_sample, np.ones((len(pts1_sample), 1))))
    l2s = (F @ pts1_homog.T).T
    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.imshow(img1_rgb)
    ax1.set_title("Image 1")
    ax1.axis("off")

    ax2.imshow(img2_rgb)
    ax2.set_title("Image 2 with Epipolar Lines")
    ax2.axis("off")

    width = image2.shape[1]

    for pt1, pt2, l2 in zip(pts1_sample, pts2_sample, l2s):
        color = tuple(np.random.rand(3))
        
        # Draw point in image 1
        ax1.scatter(pt1[0], pt1[1], color=color, edgecolors="k", zorder=5)
        
        # Draw corresponding point in image 2
        ax2.scatter(pt2[0], pt2[1], color=color, edgecolors="k", zorder=5)
        
        # Draw epipolar line in image 2: ax + by + c = 0 => y = -(ax + c) / b
        a, b, c = l2
        if b != 0:
            x_vals = np.array([0, width])
            y_vals = -(a * x_vals + c) / b
            ax2.plot(x_vals, y_vals, color=color, linewidth=1.5, alpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def analyse_image_pair(
    image1_path: Path,
    image2_path: Path,
    output_dir: Path,
    max_features: int = 4000,
    ratio: float = 0.75,
    max_image_size: int | None = 1600,
    save_figures: bool = True,
) -> PairAnalysis:
    """Run the full Week 2 analysis for one image pair.

    TODO: Complete this function by wiring together the utilities above.

    Expected steps:
    1. Load both images.
    2. Detect SIFT features.
    3. Match descriptors with Lowe's ratio test.
    4. Convert matches to point arrays.
    5. Estimate F with RANSAC.
    6. Compute epipolar errors for all filtered matches and for RANSAC inliers.
    7. Save keypoint, raw-match, filtered-match, inlier, and epipolar-line figures.
    8. Return a PairAnalysis object.
    """

    # don't need to do this? done by precompute_image_features
    # img1 = load_image(image1_path, max_image_size)
    # img2 = load_image(image2_path, max_image_size)

    features = precompute_image_features([image1_path, image2_path], max_features=max_features, max_image_size=max_image_size)

    return analyse_feature_pair(features[0], features[1], output_dir, ratio, save_figures)




def analyse_feature_pair(
    features1: ImageFeatures,
    features2: ImageFeatures,
    output_dir: Path,
    ratio: float = 0.75,
    save_figures: bool = True,
) -> PairAnalysis:
    """Run pair analysis using precomputed image features.

    TODO: Complete this function and call it from analyse_image_pair.

    This avoids recomputing SIFT features during all-pairs dataset analysis.
    In dataset mode, save_figures is normally False, so this function should
    return metrics without creating an output folder for every image pair.

    3. Match descriptors with Lowe's ratio test.
    4. Convert matches to point arrays.
    5. Estimate F with RANSAC.
    6. Compute epipolar errors for all filtered matches and for RANSAC inliers.
    7. Save keypoint, raw-match, filtered-match, inlier, and epipolar-line figures.
    8. Return a PairAnalysis object.
    """
    img1_name = features1.path.name
    img2_name = features2.path.name
    
    raw_matches = raw_descriptor_matches(features1.descriptors, features2.descriptors)
    raw_matches_count = len(raw_matches)
    filtered_matches = match_descriptors(features1.descriptors, features2.descriptors, ratio)
    
    pts1, pts2 = matched_keypoint_coords(features1.keypoints, features2.keypoints, filtered_matches)
    
    F = None
    inliers_mask = np.zeros(len(filtered_matches), dtype=bool)
    
    if len(filtered_matches) >= 8:
        F, inliers = estimate_fundamental_ransac(pts1, pts2)
        if inliers is not None:
            inliers_mask = inliers.ravel().astype(bool)
            
    inlier_count = int(np.sum(inliers_mask))
    inlier_ratio = inlier_count / len(filtered_matches) if len(filtered_matches) > 0 else 0.0
    
    mean_err_all = median_err_all = mean_err_inl = median_err_inl = max_err_inl = None
    
    if F is not None and len(filtered_matches) > 0:
        errors = compute_epipolar_errors(F, pts1, pts2)
        mean_err_all = float(np.mean(errors))
        median_err_all = float(np.median(errors))
        
        if inlier_count > 0:
            inlier_errors = errors[inliers_mask]
            mean_err_inl = float(np.mean(inlier_errors))
            median_err_inl = float(np.median(inlier_errors))
            max_err_inl = float(np.max(inlier_errors))
            
    if save_figures:
        ensure_dir(output_dir)
        draw_keypoints(features1.image, features1.keypoints, output_dir / f"keypoints_{img1_name}")
        draw_keypoints(features2.image, features2.keypoints, output_dir / f"keypoints_{img2_name}")
        
        draw_matches(features1.image, features1.keypoints, features2.image, features2.keypoints, raw_matches, output_dir / "matches_0_raw.png")
        draw_matches(features1.image, features1.keypoints, features2.image, features2.keypoints, filtered_matches, output_dir / "matches_1_filtered.png")
        
        inlier_matches = [m for m, is_inlier in zip(filtered_matches, inliers_mask) if is_inlier]
        draw_matches(features1.image, features1.keypoints, features2.image, features2.keypoints, inlier_matches, output_dir / "matches_2_inliers.png")
        
        if F is not None and inlier_count > 0:
            draw_epipolar_lines(features1.image, features2.image, pts1[inliers_mask], pts2[inliers_mask], F, output_dir / "epipolar_lines.png")
            
    return PairAnalysis(
        image_i=img1_name,
        image_j=img2_name,
        keypoints_i=len(features1.keypoints),
        keypoints_j=len(features2.keypoints),
        raw_matches=raw_matches_count,
        filtered_matches=len(filtered_matches),
        ransac_inliers=inlier_count,
        inlier_ratio=inlier_ratio,
        mean_epipolar_error_all=mean_err_all,
        median_epipolar_error_all=median_err_all,
        mean_epipolar_error_inliers=mean_err_inl,
        median_epipolar_error_inliers=median_err_inl,
        max_epipolar_error_inliers=max_err_inl,
        fundamental_matrix=F.tolist() if F is not None else None,
    )

def draw_match_graph(
    rows: list[dict],
    output_path: Path,
    min_inliers: int = 30,
) -> None:
    """Draw a match graph from pairwise metric rows.

    Edges with fewer than min_inliers are omitted to keep the graph readable.
    """
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)

    nodes = sorted({row["image_i"] for row in rows} | {row["image_j"] for row in rows})
    edges = []
    for row in rows:
        inliers = int(row["ransac_inliers"])
        if inliers >= min_inliers:
            edges.append((row["image_i"], row["image_j"], inliers))

    plt.figure(figsize=(10, 8))
    if not nodes or not edges:
        plt.text(0.5, 0.5, "No edges above threshold", ha="center", va="center")
        plt.axis("off")
    else:
        radius = 1.0
        positions = {}
        for idx, node in enumerate(nodes):
            angle = 2 * math.pi * idx / len(nodes)
            positions[node] = (radius * math.cos(angle), radius * math.sin(angle))

        max_weight = max(weight for _, _, weight in edges)
        for image_i, image_j, weight in edges:
            x1, y1 = positions[image_i]
            x2, y2 = positions[image_j]
            width = 1.0 + 4.0 * (weight / max_weight)
            plt.plot([x1, x2], [y1, y2], color="#456990", linewidth=width, alpha=0.7)
            plt.text((x1 + x2) / 2, (y1 + y2) / 2, str(weight), fontsize=7)

        for node, (x, y) in positions.items():
            plt.scatter([x], [y], s=550, color="#d8e8ff", edgecolor="#456990", zorder=3)
            label = Path(node).stem
            plt.text(x, y, label, ha="center", va="center", fontsize=8, zorder=4)

        plt.axis("off")
        plt.axis("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def select_top_initial_pairs(rows: list[dict], top_k: int = 3) -> list[dict]:
    """Select candidate Week 3 initial pairs from pairwise metrics.

    This starter version ranks by RANSAC inlier count first, then inlier ratio.
    Students should inspect the images too: the best numerical pair may have too
    little baseline for triangulation.
    """
    return sorted(
        rows,
        key=lambda row: (
            int(row["ransac_inliers"]),
            float(row["inlier_ratio"]),
            -float(row["median_epipolar_error_inliers"] or 1e9),
        ),
        reverse=True,
    )[:top_k]
