import csv
import json
from pathlib import Path

import cv2
import numpy as np


TARGET_SIZE = 300
BLACK_THRESHOLD = 128
OUTPUT_FILE = "image_data.csv"
CLASS_MAPPING_FILE = "class_mapping.txt"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
DATA_DIRECTORIES = ["circle", "square", "triangle"]


def create_headers() -> list[str]:
    """Create the class column followed by row/column pixel coordinates."""
    return [
        "object_type",
        *(
            f"({row},{column})"
            for row in range(1, TARGET_SIZE + 1)
            for column in range(1, TARGET_SIZE + 1)
        ),
    ]


def image_to_row(image_path: Path, class_id: int) -> list[int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    resized = cv2.resize(
        image,
        (TARGET_SIZE, TARGET_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    # Black pixels are 1 and white pixels are 0, as required by the dataset.
    binary = (resized < BLACK_THRESHOLD).astype(np.uint8)
    return [class_id, *binary.ravel().tolist()]


def main() -> None:
    root_directory = Path(__file__).resolve().parent
    output_path = root_directory / OUTPUT_FILE
    mapping_path = root_directory / CLASS_MAPPING_FILE
    class_directories = [root_directory / directory_name for directory_name in DATA_DIRECTORIES]
    missing_directories = [
        directory for directory in class_directories if not directory.is_dir()
    ]
    if missing_directories:
        missing_names = ", ".join(directory.name for directory in missing_directories)
        raise FileNotFoundError(f"Class folder(s) not found: {missing_names}")

    class_mapping = {
        directory.name: class_id
        for class_id, directory in enumerate(class_directories, start=1)
    }

    mapping_path.write_text(
        json.dumps(class_mapping, indent=4) + "\n",
        encoding="utf-8",
    )

    image_count = 0
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(create_headers())

        for class_directory in class_directories:
            image_paths = sorted(
                path
                for path in class_directory.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            for image_path in image_paths:
                writer.writerow(image_to_row(image_path, class_mapping[class_directory.name]))
                image_count += 1

    print(f"Created {output_path.name} with {image_count} image rows.")
    print(f"Created {mapping_path.name}: {class_mapping}")
    print(f"Columns: {TARGET_SIZE * TARGET_SIZE + 1}")


if __name__ == "__main__":
    main()