import csv
from pathlib import Path

import cv2
import numpy as np


TARGET_SIZE = 300
CSV_FILE = "image_data.csv"
OUTPUT_FILE = "validated_first_image.png"


def main() -> None:
    root_directory = Path(__file__).resolve().parent
    csv_path = root_directory / CSV_FILE
    output_path = root_directory / OUTPUT_FILE

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header row.
        first_row = next(reader, None)

    if first_row is None:
        raise ValueError(f"No image rows found in {CSV_FILE}")

    class_name = first_row[0]
    pixel_values = np.array(first_row[1:], dtype=np.uint8)

    expected_pixel_count = TARGET_SIZE * TARGET_SIZE
    if pixel_values.size != expected_pixel_count:
        raise ValueError(
            f"Expected {expected_pixel_count} pixels, found {pixel_values.size}"
        )

    # Dataset values use black=1 and white=0; PNG values use black=0 and white=255.
    image = np.where(
        pixel_values.reshape(TARGET_SIZE, TARGET_SIZE) == 1,
        0,
        255,
    ).astype(np.uint8)

    if not cv2.imwrite(str(output_path), image):
        raise IOError(f"Could not save reconstructed image: {output_path}")

    print(f"Recreated class '{class_name}' as {output_path.name}")


if __name__ == "__main__":
    main()
