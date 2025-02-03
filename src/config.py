import os
import torch
import random
import numpy as np


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed(seed)  # For CUDA-based computations
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False


BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

config = {
    'base_path': BASE_PATH,
    'data_dir': os.path.join(BASE_PATH, "data"),
    'model_dir': os.path.join(BASE_PATH, "models"),
    'sprs_v_cls_model_path': os.path.join(BASE_PATH, "models", "sparse_vector_classifier.pth"),
    'log_dir': os.path.join(BASE_PATH, "logs"),
    'job_level_data_path': os.path.join(BASE_PATH, "data", "JobLevelData.xlsx"),
    'job_level_data_file_name': "JobLevelData.xlsx",
    'model_name': 'bert-base-uncased',
    'targets_file': os.path.join(BASE_PATH, "data", 'unique_targets.txt'),
    'feature_vocab_file': 'feature2idx.pkl',
    'title_vocab_file': 'title2idx.pkl',

    'train_params': {
        'train_size': 0.8,
        'batch_size': 4,
        'epochs': 20,
        'steps_per_epoch': None,
        'latest_checkpoint_step': 50,
        'summary_step': 10,
        'max_checkpoints_to_keep': 5,
        "learning_rate": 0.001,
        "gradient_accumulation_steps": 1,
        'hidden_dim': 16,
    },
    'model_params': {
        'dropout': 0.2,
    }
}


def get_num_labels():
    # targets_file = os.path.join("data", "unique_targets.txt")
    targets_file = config['targets_file']
    if not os.path.exists(targets_file):
        raise ValueError("Targets file not found. Please run preprocessing to generate it.")
    with open(targets_file, "r") as f:
        num_labels = len(f.readlines())
    return num_labels
