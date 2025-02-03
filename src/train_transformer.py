import os
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW, get_scheduler
from sklearn.metrics import f1_score, accuracy_score

from src.model import TransformerMultiClassClassifier
from src.data import JobTitlesDataset
from src.data_preprocessing import Preprocessing
from src.config import config

# Load Configuration
batch_size = config['train_params']["batch_size"]
epochs = config['train_params']["epochs"]
learning_rate = config['train_params']["learning_rate"]
output_dir = config["model_dir"]
gradient_accumulation_steps = config['train_params']["gradient_accumulation_steps"]


def train(model, train_loader, val_loader, optimizer, scheduler, criterion, device, epochs, output_dir):
    """Training loop for the transformer model."""
    model.train()
    print("Starting training...")
    best_loss = float("inf")
    best_model_path = os.path.join(output_dir, "best_model.pth")

    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad()

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(loop):
            input_ids, attention_mask, token_type_ids, targets = batch
            input_ids, attention_mask, token_type_ids, targets = (
                input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), targets.to(device)
            )

            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            # loss = criterion(logits, targets)
            loss = criterion(logits, targets.squeeze(1))
            loss = loss / gradient_accumulation_steps  # Normalize loss for accumulated gradients
            total_loss += loss.item()

            # Backward pass
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

        # Evaluate on validation data
        val_loss, f1, acc = evaluate(model, val_loader, criterion, device)
        print(f"Validation - Loss: {val_loss:.4f}, F1 Score: {f1:.4f}, Accuracy: {acc:.4f}")

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at {best_model_path}")


def evaluate(model, data_loader, criterion, device):
    """Evaluates the model on the given dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, token_type_ids, targets = batch
            input_ids, attention_mask, token_type_ids, targets = (
                input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), targets.to(device)
            )

            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = criterion(logits, targets.squeeze(1))
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy().flatten())

    avg_loss = total_loss / len(data_loader)
    f1 = f1_score(all_targets, all_preds, average="micro")
    acc = accuracy_score(all_targets, all_preds)
    return avg_loss, f1, acc


def main():
    # Initialize Dataset and Preprocessing
    file_name = config["job_level_data_file_name"]
    dataset = JobTitlesDataset(file_name)
    processor = Preprocessing()
    processed_data = processor.preprocess(dataset)

    train_data, val_data = processed_data["train"], processed_data["val"]
    print(f"Input IDs Shape: {train_data['input_ids'].shape}")
    print(f"Attention Mask Shape: {train_data['attention_mask'].shape}")
    print(f"Token Type IDs Shape: {train_data['token_type_ids'].shape}")
    print(f"Labels Shape: {train_data['labels'].shape}")

    # Create DataLoaders
    train_dataset = TensorDataset(
        train_data["input_ids"], train_data["attention_mask"], train_data["token_type_ids"],
        train_data["labels"]
    )
    val_dataset = TensorDataset(
        val_data["input_ids"], val_data["attention_mask"], val_data["token_type_ids"],
        val_data["labels"]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Model
    model = TransformerMultiClassClassifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_loader) * epochs
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Loss Function
    criterion = torch.nn.CrossEntropyLoss()

    # Train Model
    train(model, train_loader, val_loader, optimizer, scheduler, criterion, device, epochs, output_dir)

    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
