import argparse
from model import FeatureComboMultiLabelModel


def load_model():
    """Loads the trained FeatureComboMultiLabelModel from file."""
    model = FeatureComboMultiLabelModel()
    model.load()
    return model


def process_input_features(input_list):
    """
    Dynamically assigns values to the four feature columns based on the input list.
    If more than four elements are provided, only the first four are used.

    Args:
        input_list (list): List of input feature values.

    Returns:
        dict: Processed input features mapped to the required columns.
    """
    feature_keys = ["Column 1", "Column 2", "Column 3", "Column 4"]
    processed_features = {key: "" for key in feature_keys}  # Default empty values

    for i, value in enumerate(input_list[:4]):  # Ensure max 4 elements
        processed_features[feature_keys[i]] = value

    return processed_features


def predict_job_title(input_list):
    """
    Predicts job titles based on input features.

    Args:
        input_list (list): List of input feature values.

    Returns:
        list: List of predicted job titles based on feature combinations.
    """
    model = load_model()
    input_features = process_input_features(input_list)
    predictions = model.predict(input_features)
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Inference for FeatureComboMultiLabelModel")
    parser.add_argument("--features", type=str, nargs='+', required=True,
                        help="List of feature values for inference (max 4 elements)")
    args = parser.parse_args()

    predictions = predict_job_title(args.features)

    print(f"Predicted job categories for input '{args.features}': {predictions}")


if __name__ == "__main__":
    main()
