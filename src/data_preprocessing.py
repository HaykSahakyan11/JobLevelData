import os
import torch
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from config import config


class EncodeJobTitlesDataset(Dataset):
    """
    Dataset that transforms raw job titles and features into encoded torch tensors.
    For multi-class classification, the target is returned as an integer index.
    """

    def __init__(self, raw_dataset, feature2idx, title2idx):
        self.raw_dataset = raw_dataset
        self.feature2idx = feature2idx
        self.title2idx = title2idx

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        title, features = self.raw_dataset[idx]
        feature_vec = encode_features(features, self.feature2idx)
        target_idx = encode_title(title, self.title2idx)
        # Convert to torch tensors; target as LongTensor
        feature_tensor = torch.tensor(feature_vec, dtype=torch.float32)
        target_tensor = torch.tensor(target_idx, dtype=torch.long)
        return feature_tensor, target_tensor


def encode_features(features, feature2idx):
    """
    Convert a list of feature strings into a multi-hot vector.

    Args:
        features (list of str): Feature values for a data point.
        feature2idx (dict): Mapping of feature value to index.

    Returns:
        np.array: Multi-hot encoded vector of shape (len(feature2idx),)
    """
    vec = np.zeros(len(feature2idx), dtype=np.float32)
    for f in features:
        f = f.strip()
        if f and f in feature2idx:
            vec[feature2idx[f]] = 1.0
    return vec


def encode_title(title, title2idx):
    """
    Convert a title string into a class index.

    Args:
        title (str): The job title.
        title2idx (dict): Mapping of title to index.

    Returns:
        int: The index corresponding to the job title.
    """
    return title2idx.get(title, -1)  # returns -1 if title not found


def build_vocabs(dataset):
    """
    Build vocabularies for feature values and job titles.

    Args:
        dataset (JobTitlesDataset): The raw dataset.

    Returns:
        feature2idx (dict): Mapping from feature value to index.
        title2idx (dict): Mapping from title to index.
    """
    feature_set = set()
    title_set = set()
    for title, features in dataset:
        title_set.add(title)
        for f in features:
            f = f.strip()
            if f:
                feature_set.add(f)
    feature2idx = {feat: i for i, feat in enumerate(sorted(feature_set))}
    title2idx = {title: i for i, title in enumerate(sorted(title_set))}
    return feature2idx, title2idx


def save_vocab(vocab, file_name):
    """Save vocabulary dictionary to a pickle file."""
    file_path = os.path.join(config['data_dir'], file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Saved vocab to {file_path}")


def load_vocab(file_name):
    """Load vocabulary dictionary from a pickle file."""
    file_path = os.path.join(config['data_dir'], file_name)
    with open(file_path, 'rb') as f:
        vocab = pickle.load(f)
    print(f"Loaded vocab from {file_path}")
    return vocab


class Preprocessing:
    """
    Handles data preprocessing for generating job titles from labels.
    Converts label columns into tokenized inputs and job titles into multi-hot encoded labels.
    """

    def __init__(self, tokenizer_name="bert-base-uncased", max_length=64, train_size=0.8):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.train_size = train_size
        # self.mlb = LabelEncoder()  # Multi-hot encoding for large label sets
        self.label_encoder = LabelEncoder()

    def load_data(self, dataset):
        """ Loads raw labels (features) and titles (targets) from the dataset. """
        labels, titles = dataset.get_raw_data()
        combined_labels = [list(filter(None, label_row)) for label_row in labels]  # Remove empty labels
        return combined_labels, titles

    def tokenize_inputs(self, inputs):
        """ Tokenizes label strings using a BERT tokenizer. """
        tokenized = self.tokenizer(
            inputs,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True
        )
        return tokenized["input_ids"], tokenized["attention_mask"], tokenized["token_type_ids"]

    def encode_labels(self, labels):
        """ Encodes labels into integers for multi-class classification. """

        self.label_encoder.fit(labels)  # Fit on all available labels
        encoded_labels = self.label_encoder.transform(labels)
        self.save_label_encoder()
        return torch.tensor(encoded_labels, dtype=torch.long).unsqueeze(1)

    def save_label_encoder(self):
        """Save LabelEncoder to a file for inference."""
        data_path = config['data_dir']
        file_path = os.path.join(data_path, "label_encoder.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(self.label_encoder, f)

    def preprocess(self, dataset):
        """ Full pipeline: tokenizes inputs (labels), encodes labels, and splits data. """
        combined_labels, titles = self.load_data(dataset)

        # Convert labels to multi-hot encoded format
        encoded_labels = self.encode_labels(titles)

        # Tokenize inputs (labels as text)
        formatted_labels = [", ".join(lbl) for lbl in combined_labels]
        input_ids, attention_mask, token_type_ids = self.tokenize_inputs(formatted_labels)

        # Split data into training and validation sets
        (train_input_ids, val_input_ids,
         train_labels, val_labels,
         train_attention_mask, val_attention_mask,
         train_token_type_ids, val_token_type_ids,
         # train_formatted_labels, val_formatted_labels
         ) = train_test_split(
            input_ids, encoded_labels, attention_mask, token_type_ids,
            train_size=self.train_size, random_state=42
        )

        return {
            "train": {
                "input_ids": train_input_ids,
                "attention_mask": train_attention_mask,
                "token_type_ids": train_token_type_ids,
                "labels": train_labels,
            },
            "val": {
                "input_ids": val_input_ids,
                "attention_mask": val_attention_mask,
                "token_type_ids": val_token_type_ids,
                "labels": val_labels,
            }
        }


def load_label_encoder():
    """Loads the saved LabelEncoder for decoding predictions."""
    data_path = config['data_dir']
    file_path = os.path.join(data_path, "label_encoder.pkl")
    with open(file_path, "rb") as f:
        label_encoder = pickle.load(f)
    return label_encoder


if __name__ == "__main__":
    """
    Testing Preprocessing when executed as a standalone script.
    """
    from data import JobTitlesDataset

    file_name = "JobLevelData.xlsx"
    dataset = JobTitlesDataset(file_name)
    processor = Preprocessing()
    processed_data = processor.preprocess(dataset)

    train_data, val_data = processed_data["train"], processed_data["val"]

    print("First Tokenized Train Input IDs:", train_data["input_ids"][0])
    print("First Attention Mask Train Sample:", train_data["attention_mask"][0])
    print("First Token Type IDs Train Sample:", train_data["token_type_ids"][0])
    print("First Encoded Train Target Sample:", train_data["labels"][0])
