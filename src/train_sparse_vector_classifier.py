import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from src.config import set_seed, config
from src.data import JobTitlesDataset
from src.data_preprocessing import EncodeJobTitlesDataset, build_vocabs, save_vocab, load_vocab
from src.model import SparseVectorClassifier


class ModelTrainer:
    """
    Trainer class that encapsulates training, evaluation, prediction decoding, and best model saving.
    """

    def __init__(self, model, train_loader, criterion, optimizer, device, title2idx):
        """
        Initializes the trainer.

        Args:
            model (nn.Module): The neural network model.
            train_loader (DataLoader): DataLoader for training data.
            criterion (loss function): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer.
            device (torch.device): Device on which to train.
            title2idx (dict): Vocabulary mapping for titles.
        """
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model.to(self.device)
        self.title2idx = title2idx
        # Build reverse mapping for decoding predictions
        self.idx2title = {i: title for title, i in title2idx.items()}
        self.best_accuracy = 0.0
        self.best_epoch = -1

    def train(self, num_epochs, eval_loader=None, save_dir="models"):
        """
        Train the model and evaluate at the end of each epoch. The best performing model is saved.

        Args:
            num_epochs (int): Number of training epochs.
            eval_loader (DataLoader): DataLoader for evaluation (validation) data.
                                      If None, evaluation is skipped.
            save_dir (str): Directory where the best model will be saved.
        """
        print("Starting training...")
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            for features, targets in self.train_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(features)
                loss = self.criterion(logits, targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            avg_train_loss = epoch_loss / len(self.train_loader)

            # If an evaluation loader is provided, run evaluation and save best model.
            if eval_loader is not None:
                eval_loss, accuracy = self.evaluate(eval_loader)
                print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} "
                      f"- Val Loss: {eval_loss:.4f} - Accuracy: {accuracy * 100:.2f}%")
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_epoch = epoch + 1
                    self.save_model(save_dir, model_name=f"best_model_epoch_{epoch + 1}_acc_{accuracy * 100:.2f}.pt")
            else:
                print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")

        if eval_loader is not None:
            print(f"Best model saved from epoch {self.best_epoch} with accuracy {self.best_accuracy * 100:.2f}%")

    def evaluate(self, eval_loader):
        """
        Evaluate the model on the evaluation dataset.

        Args:
            eval_loader (DataLoader): DataLoader for evaluation data.

        Returns:
            Tuple: (average evaluation loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, targets in eval_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                logits = self.model(features)
                loss = self.criterion(logits, targets)
                total_loss += loss.item()
                predicted = torch.argmax(logits, dim=1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
        avg_loss = total_loss / len(eval_loader)
        accuracy = correct / total if total > 0 else 0
        return avg_loss, accuracy

    def decode_prediction(self, logits):
        """
        Decodes the prediction logits to a predicted class.

        Args:
            logits (torch.Tensor): Logits output from the model (1D tensor).

        Returns:
            dict: A dictionary with 'predicted_index' and 'predicted_title'.
        """
        predicted_idx = torch.argmax(logits).item()
        predicted_title = self.idx2title.get(predicted_idx, "Unknown")
        return {'predicted_index': predicted_idx, 'predicted_title': predicted_title}

    def evaluate_sample(self, sample_features):
        """
        Evaluate a single sample and decode its prediction.

        Args:
            sample_features (torch.Tensor): Input feature tensor of shape (input_dim,).

        Returns:
            dict: Decoded prediction.
        """
        self.model.eval()
        sample_features = sample_features.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(sample_features).squeeze()
        return self.decode_prediction(logits)

    def save_model(self, save_dir, model_name="sparse_vector_classifier.pt"):
        """
        Save the current model to the specified directory.

        Args:
            save_dir (str): Directory where the model should be saved.
            model_name (str): Name of the saved model file.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, model_name)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


def main():
    # Set random seed for reproducibility
    set_seed(42)

    # Load raw dataset from Excel
    file_name = config["job_level_data_file_name"]
    dataset = JobTitlesDataset(file_name)

    # Build and save vocabularies
    feature2idx, title2idx = build_vocabs(dataset)
    save_vocab(feature2idx, config['feature_vocab_file'])
    save_vocab(title2idx, config['title_vocab_file'])

    # Create encoded dataset
    encoded_dataset = EncodeJobTitlesDataset(raw_dataset=dataset, feature2idx=feature2idx, title2idx=title2idx)

    # Create DataLoader for training and evaluation.
    train_loader = DataLoader(encoded_dataset, batch_size=config['train_params']['batch_size'], shuffle=True)
    eval_loader = DataLoader(encoded_dataset, batch_size=config['train_params']['batch_size'], shuffle=False)

    # Model parameters
    input_dim = len(feature2idx)
    num_classes = len(title2idx)
    hidden_dim = config['train_params']['hidden_dim']

    # Initialize model
    model = SparseVectorClassifier(input_dim, hidden_dim, num_classes)
    print("Model architecture:")
    print(model)

    # Define loss function and optimizer.
    criterion = nn.CrossEntropyLoss()  # Using CrossEntropyLoss since targets are class indices.
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train_params']['learning_rate'])

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize trainer with the title vocabulary for decoding predictions
    trainer = ModelTrainer(model, train_loader, criterion, optimizer, device, title2idx)

    # Train the model with evaluation and callback to save the best model.
    trainer.train(config['train_params']['epochs'], eval_loader=eval_loader, save_dir="models")

    # Evaluate on a sample from the dataset (for example, the first sample)
    sample_features, sample_target = encoded_dataset[0]
    decoded_pred = trainer.evaluate_sample(sample_features)

    print("Sample target (class index):", sample_target.item())
    print("Decoded prediction:", decoded_pred)


if __name__ == "__main__":
    main()
