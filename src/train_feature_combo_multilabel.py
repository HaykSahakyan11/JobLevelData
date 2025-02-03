import os
import pickle
import pandas as pd
from src.config import config
from model import FeatureComboMultiLabelModel


def main():
    """
    Train and save the FeatureComboMultiLabelModel externally.
    """
    # Load dataset
    file_name = config["job_level_data_file_name"]
    data_path = os.path.join(config['data_dir'], file_name)
    dataset = pd.read_excel(data_path)

    # Initialize and train the model
    model = FeatureComboMultiLabelModel()
    model.train(dataset)

    # Save the trained model
    model.save()
    print("FeatureComboMultiLabelModel training completed and saved successfully.")


if __name__ == "__main__":
    main()