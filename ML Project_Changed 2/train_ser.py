import sys
import argparse
import pickle
import torch
import numpy as np
import os
import random
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import time
import timm
from torch.backends import cudnn
from data_utils import SERDataset
from models.ser_model import Ser_Model
from torchvision import transforms
from PIL import Image


# Visualizations imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

# Color mapping for visualization
colors_per_class = {
    0: [0, 0, 0],
    1: [255, 107, 107],
    2: [100, 100, 255],
    3: [16, 172, 132],
}

from torchvision import transforms

def main(args):
    # Aggregate parameters
    num_classes = 4  # Add the correct number of classes for your dataset
    params = {
        'ser_task': 'SLM',
        'repeat_idx': args.repeat_idx,
        'val_id': args.val_id,
        'test_id': args.test_id,
        'num_epochs': args.num_epochs,
        'early_stop': args.early_stop,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'random_seed': args.seed,
        'use_gpu': args.gpu,
        'gpu_ids': args.gpu_ids,
        'save_label': args.save_label,
        'oversampling': args.oversampling,
        'pretrained': args.pretrained,
        'num_classes': num_classes  # Add num_classes to params
    }

    print('*' * 40)
    print(f"\nPARAMETERS:\n")
    print('*' * 40)
    for key in params:
        print(f'{key:>15}: {params[key]}')
    print('*' * 40)
    
    # Set random seed
    seed_everything(params['random_seed'])

    # Load dataset
    num_entries = 5
    with open(args.features_file, "rb") as fin:
        features_data = pickle.load(fin)

        if isinstance(features_data, list):  # For list-based datasets
            features_data = features_data[:num_entries]
        elif isinstance(features_data, dict):  # For dictionary-based datasets
            features_data = {k: features_data[k] for i, k in enumerate(features_data) if i < num_entries}
        else:
            raise ValueError("Unsupported data format in the .pkl file")
    


    # Add resize transformation to the dataset initialization
    transform = transforms.Compose([
        transforms.Lambda(lambda x: Image.fromarray(np.uint8(np.squeeze(x) * 255)) if isinstance(x, np.ndarray) else x),  # Convert numpy array to PIL Image
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB (3 channels)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard RGB normalization
    ])


    ser_dataset = SERDataset(features_data, 
                              val_speaker_id=args.val_id,
                              test_speaker_id=args.test_id,
                              oversample=args.oversampling,
                              transform=transform)  # Apply the transformation

    # Train model
    train_stat = train(ser_dataset, params, save_label=args.save_label)
    return train_stat


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train a SER model in an iterative-based manner with PyTorch and IEMOCAP dataset.")

    # Features
    parser.add_argument('features_file', type=str,
                        help='Features extracted from `extract_features.py`.')

    # Training
    parser.add_argument('--repeat_idx', type=str, default='0', help='ID of repeat_idx')
    parser.add_argument('--val_id', type=str, default='1F', help='ID of speaker to be used as validation')
    parser.add_argument('--test_id', type=str, default='1M', help='ID of speaker to be used as test')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs.')
    parser.add_argument('--early_stop', type=int, default=4, help='Number of early stopping epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--seed', type=int, default=100, help='Random seed for reproducibility.')
    parser.add_argument('--gpu', type=int, default=1, help='If 1, use GPU')
    parser.add_argument('--gpu_ids', type=list, default=[0], help='List of GPU ids to use.')

    # Best Model
    parser.add_argument('--save_label', type=str, default=None,
                        help='Label for the current run, used to save the best model')

    # Parameters for model tuning
    parser.add_argument('--oversampling', action='store_true',
                        help='If true, apply random oversampling to balance training dataset')
    
    parser.add_argument('--pretrained', action='store_true',
                        help='If set, use pre-trained weights for the model.')

    return parser.parse_args(argv)


def train(dataset, params, save_label='default'):
    # Prepare dataset
    train_dataset = dataset.get_train_dataset()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

    val_dataset = dataset.get_val_dataset()
    test_dataset = dataset.get_test_dataset()

    print("Using transformer model")

    # Select device
    device = torch.device("cuda:0" if params['use_gpu'] == 1 else "cpu")

    # Initialize the transformer model
    model = SerModelTransformer(num_classes=params['num_classes']).to(device)

    print(f"Model initialized: {model}")
    print(f"Number of trainable parameters: {count_parameters(model)}")

    # Set loss criterion and optimizer
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'])
    criterion_ce = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = -1e8
    best_val_loss = 1e8
    for epoch in range(params['num_epochs']):
        model.train()
        total_loss = 0
        for train_batch in tqdm(train_loader):
            # Get data
            inputs = train_batch['seg_spec'].to(device)  # Adjust for your dataset
            labels = train_batch['seg_label'].to(device, dtype=torch.long)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)

            # Compute loss
            loss = criterion_ce(outputs, labels)
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{params['num_epochs']} - Loss: {total_loss / len(train_loader)}")

        # Validation
        val_stat = test('VALIDATION', params, model, criterion_ce, val_dataset, params['batch_size'], device)
        print(f"Validation loss: {val_stat[0]} | Weighted Accuracy: {val_stat[1]}% | Unweighted Accuracy: {val_stat[2]}%")

        # Early stopping
        if val_stat[1] > best_val_acc:
            best_val_acc = val_stat[1]
            best_val_loss = val_stat[0]
            if save_label:
                torch.save(model.state_dict(), f'best_model_{save_label}.pth')

    return {'best_val_acc': best_val_acc, 'best_val_loss': best_val_loss}



def test(mode, params, model, criterion_ce, test_dataset, batch_size, device, return_matrix=False):
    model.eval()
    total_loss = 0
    test_preds = []
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for test_batch in tqdm(test_loader):
        test_data = test_batch['seg_spec'].to(device)
        test_labels = test_batch['seg_label'].to(device, dtype=torch.long)

        with torch.no_grad():
            test_outputs = model(test_data)
            loss = criterion_ce(test_outputs, test_labels)
            total_loss += loss.item()

            preds = torch.argmax(test_outputs, dim=1)
            test_preds.append(preds.cpu().numpy())

    test_preds = np.concatenate(test_preds, axis=0)
    test_loss = total_loss / len(test_loader)

    # Accuracy metrics
    test_wa = weighted_accuracy(test_preds, test_dataset)
    test_ua = unweighted_accuracy(test_preds, test_dataset)

    results = (test_loss, test_wa * 100, test_ua * 100)
    
    if return_matrix:
        test_conf = confusion_matrix(test_preds, test_dataset)
        return results, test_conf
    else:
        return results


def weighted_accuracy(preds, dataset):
    # Calculate weighted accuracy
    return dataset.weighted_accuracy(preds)


def unweighted_accuracy(preds, dataset):
    # Calculate unweighted accuracy
    return dataset.unweighted_accuracy(preds)


def confusion_matrix(preds, dataset):
    return dataset.confusion_matrix(preds)


class SerModelTransformer(nn.Module):
    def __init__(self, num_classes=6):
        super(SerModelTransformer, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(args)
