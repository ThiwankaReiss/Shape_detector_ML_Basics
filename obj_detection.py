import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


TARGET_SIZE = 300
BLACK_THRESHOLD = 128
CSV_FILE = "image_data.csv"
CLASS_MAPPING_FILE = "class_mapping.txt"
IMAGE_FILE = "test_image.png"


def image_to_features(image_path: Path) -> np.ndarray:
	image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
	if image is None:
		raise ValueError(f"Could not read image: {image_path}")

	resized = cv2.resize(
		image,
		(TARGET_SIZE, TARGET_SIZE),
		interpolation=cv2.INTER_AREA,
	)
	binary = (resized < BLACK_THRESHOLD).astype(np.uint8)
	return binary.reshape(1, -1)


def load_class_mapping(mapping_path: Path) -> dict[str, int]:
	with mapping_path.open(encoding="utf-8") as mapping_file:
		mapping = json.load(mapping_file)
	return {name: int(class_id) for name, class_id in mapping.items()}


def predict_image(
	model: DecisionTreeRegressor,
	image_path: Path,
	class_mapping: dict[str, int],
) -> None:
	image_features = pd.DataFrame(
		image_to_features(image_path),
		columns=model.feature_names_in_,
	)
	predicted_id = float(model.predict(image_features)[0])
	valid_ids = sorted(class_mapping.values())
	class_id = min(valid_ids, key=lambda value: abs(value - predicted_id))
	class_name = next(
		name for name, value in class_mapping.items() if value == class_id
	)
	uncertainty = abs(predicted_id - class_id) * 100 / class_id

	print(f"Predicted object type value: {predicted_id:.2f}")
	print(f"Predicted object: {class_name} (ID {class_id})")
	print(f"Uncertainty: {uncertainty:.2f}%")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Train an object model and classify one input image."
	)
	parser.add_argument(
		"image_path",
		type=Path,
		nargs="?",
		default=IMAGE_FILE,
		help=f"Path to the image that should be classified (default: {IMAGE_FILE})",
	)
	args = parser.parse_args()

	root_directory = Path(__file__).resolve().parent
	data = pd.read_csv(root_directory / CSV_FILE)
	class_mapping = load_class_mapping(root_directory / CLASS_MAPPING_FILE)

	y = data["object_type"]
	X = data.drop(columns=["object_type"])

	train_X, val_X, train_y, val_y = train_test_split(
		X,
		y,
		test_size=0.25,
		random_state=1,
		stratify=y,
	)

	model = DecisionTreeRegressor(random_state=1)
	model.fit(train_X, train_y)
	val_predictions = model.predict(val_X)
	val_mae = mean_absolute_error(val_y, val_predictions)
	print(f"Validation MAE: {val_mae:.2f}")

	if not args.image_path.is_file():
		raise FileNotFoundError(f"Input image does not exist: {args.image_path}")

	predict_image(model, args.image_path, class_mapping)


if __name__ == "__main__":
	main()
