import torch
import argparse
from src.config import config, set_seed
from src.data_preprocessing import encode_features, load_vocab
from src.model import SparseVectorClassifier


def load_model(model_path, input_dim, hidden_dim, num_classes, device):
    """
    Loads the model architecture and weights.
    """
    model = SparseVectorClassifier(input_dim, hidden_dim, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_vocabularies():
    """
    Loads the saved feature and title vocabularies.
    """
    feature2idx = load_vocab(config['feature_vocab_file'])
    title2idx = load_vocab(config['title_vocab_file'])
    return feature2idx, title2idx


def predict(sample_features):
    """
    Predicts the class for a given set of job labels.
    """
    feature2idx, title2idx = load_vocabularies()
    feature_vec = encode_features(sample_features, feature2idx)
    feature_tensor = torch.tensor(feature_vec, dtype=torch.float32).unsqueeze(0)

    input_dim = len(feature2idx)
    num_classes = len(title2idx)
    hidden_dim = config['train_params']['hidden_dim']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = config['sprs_v_cls_model_path']
    model = load_model(model_path, input_dim, hidden_dim, num_classes, device)

    feature_tensor = feature_tensor.to(device)
    with torch.no_grad():
        logits = model(feature_tensor).squeeze()
    predicted_idx = torch.argmax(logits).item()

    idx2title = {i: title for title, i in title2idx.items()}
    predicted_title = idx2title.get(predicted_idx, "Unknown")
    return predicted_idx, predicted_title


def main():
    parser = argparse.ArgumentParser(description="Inference for Sparse Vector Classifier")
    parser.add_argument("--job_label", type=str, nargs='+', required=True, help="List of job labels for inference")
    args = parser.parse_args()

    predicted_idx, predicted_title = predict(args.job_label)
    print(f"Predicted class index: {predicted_idx}")
    print(f"Predicted title: {predicted_title}")


if __name__ == "__main__":
    set_seed(42)
    main()
