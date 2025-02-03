import os
import torch
import argparse
from transformers import AutoTokenizer
from src.model import TransformerMultiClassClassifier
from src.config import config
from src.data_preprocessing import load_label_encoder

# Load Configuration
model_path = os.path.join(config["model_dir"], "best_model.pth")
model_name = config["model_name"]
tokenizer = AutoTokenizer.from_pretrained(model_name)


def load_model():
    """Loads the trained Transformer model for inference."""
    model = TransformerMultiClassClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def predict(job_labels, model, tokenizer):
    """Predicts the class labels for given job titles."""
    inputs = tokenizer(
        job_labels,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
        return_token_type_ids=True,
        return_attention_mask=True
    )

    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"]
        )
        predicted_classes = torch.argmax(logits, dim=1).tolist()

    return predicted_classes


def main():
    parser = argparse.ArgumentParser(description="Inference for Transformer-based Job Title Classification")
    parser.add_argument("--job_label", type=str, nargs='+', required=True, help="List of job titles for inference")
    args = parser.parse_args()

    model = load_model()
    label_encoder = load_label_encoder()
    predicted_class_indices = predict(args.job_label, model, tokenizer)
    predicted_labels = label_encoder.inverse_transform(predicted_class_indices)

    print(f"Predicted Classes for input '{args.job_label}': {predicted_labels}")


if __name__ == "__main__":
    main()
