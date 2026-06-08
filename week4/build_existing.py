import argparse
from pathlib import Path
import open3d as o3d

def load_ply_files(path: Path):
    if path.is_dir():
        return sorted(path.glob("*.ply"))
    return [path]

def main():
    parser = argparse.ArgumentParser(description="Visualize PLY file(s) with Open3D")
    parser.add_argument("path", type=Path, help="PLY file or directory containing PLY files")
    args = parser.parse_args()

    ply_paths = load_ply_files(args.path)
    if not ply_paths:
        raise SystemExit(f"No PLY files found at {args.path}")

    geometries = []
    for ply_path in ply_paths:
        mesh = o3d.io.read_point_cloud(str(ply_path))
        if mesh.is_empty():
            print(f"Warning: {ply_path} is empty or failed to load")
            continue
        geometries.append(mesh)

    if not geometries:
        raise SystemExit("No valid PLY geometries to display.")

    o3d.visualization.draw_geometries(geometries, window_name="PLY Viewer")

if __name__ == "__main__":
    main()