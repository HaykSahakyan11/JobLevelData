import os
import pandas as pd

from torch.utils.data import Dataset

from src.config import config


class JobTitlesDataset(Dataset):
    """
    Custom PyTorch dataset for multi-label classification of job titles.
    Loads data from an Excel file and provides raw titles and labels.
    """

    def __init__(self, file_name=None):
        if file_name is None:
            data_path = config['job_level_data']
        else:
            data_path = os.path.join(config['data_dir'], file_name)
        self.df = pd.read_excel(data_path)
        self.titles = self.df["Title"].astype(str).tolist()
        self.labels = self.df[['Column 1', 'Column 2', 'Column 3', 'Column 4']].fillna('').values.tolist()

    def get_raw_data(self):
        """ Returns raw job titles and labels. """
        return self.labels, self.titles

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        return self.titles[idx], self.labels[idx]


if __name__ == "__main__":
    """
    Testing JobTitlesDataset when executed as a standalone script.
    """
    file_name = "JobLevelData.xlsx"
    dataset = JobTitlesDataset(file_name)
    print("Dataset Test: First Item Sample")
    print(dataset[0])
