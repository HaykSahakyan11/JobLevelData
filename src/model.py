import os
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoConfig

from config import config, get_num_labels


class TransformerMultiClassClassifier(nn.Module):
    """
    Transformer-based Multi-Label Classification Model.
    Uses a pretrained transformer model as the backbone and adds a classification head.
    """

    def __init__(self):
        super(TransformerMultiClassClassifier, self).__init__()

        num_labels = get_num_labels()
        model_name = config['model_name']

        # Load Transformer Model Configuration
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)

        # Load Pretrained Transformer Model
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)

        # Dropout Layer
        self.dropout = nn.Dropout(config['model_params']['dropout'])

        # Classification Head
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """Forward pass of the model."""

        # Transformer Model Forward Pass
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Extract CLS Token Output
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Apply Dropout
        cls_output = self.dropout(cls_output)

        # Classification Head
        logits = self.classifier(cls_output)

        return self.softmax(logits)


class SparseVectorClassifier(nn.Module):
    """
    Simple feedforward neural network for multi-class classification.
    """

    def __init__(self, input_dim, hidden_dim, num_classes):
        """
        Initializes the model.

        Args:
            input_dim (int): Dimensionality of the input (number of features).
            hidden_dim (int): Number of neurons in the hidden layer.
            num_classes (int): Number of output classes.
        """
        super(SparseVectorClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Logits for each class (no softmax applied)
        """
        x = F.relu(self.fc1(x))
        # No softmax here because CrossEntropyLoss expects raw logits.
        logits = self.fc2(x)
        return logits


class FeatureComboMultiLabelModel:
    """
    A multi-label classification model that predicts job titles based on
    a unique combination of features from given columns.

    The model groups the training data based on non-null feature combinations
    and stores a lookup dictionary mapping these combinations to a list of job titles.
    """

    def __init__(self):
        # Dictionary that maps feature combination string -> array of titles
        self.grouped_titles = {}
        self.feature_cols = ['Column 1', 'Column 2', 'Column 3', 'Column 4']

    def _create_feature_combination(self, row):
        """
        Create a feature combination string for a row based on non-null values.
        Sorting ensures the order is consistent.
        """
        # Only include non-null, non-empty values
        features = [str(row[col]) for col in self.feature_cols if pd.notnull(row[col]) and row[col]]
        return '-'.join(sorted(features))

    def train(self, df):
        """
        Train the model by building a lookup dictionary of feature combinations to job titles.

        Parameters:
            df (pd.DataFrame): Training dataframe which must include the feature columns
                               and a 'Title' column.
        """
        # Create a new column in the dataframe that represents the feature combination.
        df['Feature Combination'] = df.apply(self._create_feature_combination, axis=1)

        # Group by the feature combination and collect unique job titles per group.
        self.grouped_titles = (
            df.groupby('Feature Combination')['Title']
            .unique()
            .to_dict()
        )
        print("Training completed. Number of unique feature combinations:", len(self.grouped_titles))

    def predict(self, input_features):
        """
        Predict job titles for a given input based on the learned lookup dictionary.

        Parameters:
            input_features (dict): A dictionary containing the feature values for
                                   keys 'Column 1', 'Column 2', 'Column 3', 'Column 4'

        Returns:
            list: A list of predicted job titles that belong to the same feature combination.
        """
        # Create feature combination from the input features.
        feature_comb = '-'.join(
            [str(input_features[col]) for col in self.feature_cols
             if bool(pd.notnull(input_features[col]) and input_features[col])]
        )
        # Retrieve and return the list of job titles for that feature combination.
        return list(self.grouped_titles.get(feature_comb, []))

    def save(self):
        """
        Save the trained model (lookup dictionary) to a file.

        Parameters:
            filepath (str): The path to the file where the model should be saved.
        """
        data_path = config['data_dir']
        filepath = os.path.join(data_path, 'grouped_titles')
        with open(filepath, 'wb') as f:
            pickle.dump({
                'grouped_titles': self.grouped_titles,
                'feature_cols': self.feature_cols
            }, f)
        print(f"Model saved to {filepath}")

    def load(self):
        """
        Load a trained model (lookup dictionary) from a file.

        Parameters:
            filepath (str): The path to the file where the model is saved.
        """
        data_path = config['data_dir']
        filepath = os.path.join(data_path, 'grouped_titles')
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.grouped_titles = data.get('grouped_titles', {})
                self.feature_cols = data.get('feature_cols', self.feature_cols)
            print(f"Model loaded from {filepath}")
        else:
            print("File does not exist. Please check the filepath.")
