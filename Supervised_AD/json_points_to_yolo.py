from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_CLASS_MAP = {
    "ColorStreak": 0,
    "Hole": 1,
    "Splice": 2,
    "Stain": 3,
    "Uneven": 4,
    "Wrinkles": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LabelMe-style JSON points to YOLO bbox txt.")
    parser.add_argument("--json", type=str, help="Single json file to convert.")
    parser.add_argument("--json-dir", type=str, help="Directory containing json files to convert.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output txt directory.")
    return parser.parse_args()


def points_to_bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_to_yolo(
    xmin: float, ymin: float, xmax: float, ymax: float, image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    x_center = ((xmin + xmax) / 2.0) / image_width
    y_center = ((ymin + ymax) / 2.0) / image_height
    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height
    return x_center, y_center, width, height


def convert_one(json_path: Path, out_dir: Path) -> Path:
    data = json.loads(json_path.read_text())
    image_width = int(data["imageWidth"])
    image_height = int(data["imageHeight"])
    rows: list[str] = []

    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label not in DEFAULT_CLASS_MAP:
            raise ValueError(f"Unknown label '{label}' in {json_path}")

        points = shape.get("points", [])
        if len(points) < 2:
            continue

        xmin, ymin, xmax, ymax = points_to_bbox(points)
        x_center, y_center, width, height = bbox_to_yolo(
            xmin, ymin, xmax, ymax, image_width, image_height
        )
        rows.append(
            f"{DEFAULT_CLASS_MAP[label]} "
            f"{x_center:.12f} {y_center:.12f} {width:.12f} {height:.12f}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{json_path.stem}.txt"
    out_path.write_text("\n".join(rows) + ("\n" if rows else ""))
    return out_path


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    if args.json:
        json_files = [Path(args.json)]
    elif args.json_dir:
        json_files = sorted(Path(args.json_dir).glob("*.json"))
    else:
        raise ValueError("Either --json or --json-dir must be provided.")

    for json_file in json_files:
        out_path = convert_one(json_file, out_dir)
        print(f"{json_file} -> {out_path}")


if __name__ == "__main__":
    main()
