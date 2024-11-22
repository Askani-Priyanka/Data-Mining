#imports
import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from hashlib import md5
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(loader, model, device):
    """
    Evaluate the model on a given data loader.

    Args:
    loader (torch.utils.data.DataLoader): The data loader containing the dataset to evaluate.

    Returns:
    tuple: A tuple containing the following metrics:
      - accuracy (float): The percentage of correctly classified samples.
      - precision: float. The weighted average precision score across all classes.
      - recall (float): The weighted average recall score across all classes.
      - f1 (float): The weighted average F1 score across all classes.

    Description:
      1. This function sets the model to evaluation mode-deactivating both dropout and updating batch normalization statistic.
      2. Iterates through the dataset and computes the following key evaluation metrics: accuracy, precision, recall, and F1 score.
      3. These metrics are computed comparing the predictions with true targets.
      4.  All computations without gradient updates(`torch.no_grad()`), ensuring that the evaluation process does not affect the model.

      Steps:
      1. Pass each batch through the model to get predictions.
      2. Compare predictions with ground truth to calculate the number of correct predictions.
      3. Collect all predictions and true labels for calculating precision, recall, and F1 scores.
      4. Return the computed metrics for analysis.
    """
    model.eval()
    correct, total = 0, 0
    all_targets, all_preds = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = 100. * correct / total
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    cm = confusion_matrix(all_targets, all_preds)

    return accuracy, precision, recall, f1, cm

def plot_confusion_matrix(cm, class_names):
    """
    Plot the confusion matrix using Seaborn's heatmap.
    """
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, linewidths=1, linecolor='black')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.jpeg')
    plt.show()

# Load CIFAR-10 test data
transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):

        """
        Initialize a residual block for ResNet.

        Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride for the first convolution layer. Default is 1.

        Description:
        Define a Residual Block for the ResNet model
        It consists of two convolutional layers, batch normalization, ReLU activation, and a skip connection.
        Skip connectivity allows gradients to go forward directly into earlier layers and
        prevents vanishing gradient problems.
        """
        super(ResidualBlock, self).__init__()
        # First convolutional layer in the block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolutional layer in the block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Dropout is added with a rate of 50% to improve regularization and reduce overfitting
        self.dropout = nn.Dropout(0.5)

        # Define the skip connection. If the input and output dimensions differ, use a 1x1 convolution to match them.
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        Forward pass through the residual block.

        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
        torch.Tensor: Output tensor after applying the residual block.
        """
        # Save the original input for the skip connection
        identity = self.skip_connection(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply dropout after the second convolution
        out = self.dropout(out)

        # Add the skip connection
        out += identity

        # Apply ReLU activation to the result
        return self.relu(out)

# Define the ResNet-34 model using the Residual Blocks
class ResNet34(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        """
        Initialize the ResNet-34 architecture.

        Args:
        block (nn.Module): Instance of Residual block class.
        num_blocks (list): Number of residual blocks for each layer group.
        num_classes (int, optional): Number of output classes for classification. Default is 10.

        Description:
        ResNet-34 is a deep convolutional neural network architecture comprising an initial convolutional layer,
        4 groups of residual layers, and a fully connected layer. In this model, skip connections are used to
        improve gradient flow and stability during training.
        """
        super(ResNet34, self).__init__()
        self.in_channels = 64  # Initial number of channels for the input layer

        # First convolutional layer for preprocessing
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Define the layers of the ResNet model
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)   # First group of residual blocks
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # Second group with downsampling
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # Third group with more channels
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # Fourth group with the highest channels

        # Global average pooling and a fully connected layer for classification
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        """
        Create a layer group consisting of multiple residual blocks.

        Args:
        block (nn.Module): Residual block class to use.
        out_channels (int): Number of output channels for the blocks in this layer group.
        blocks (int): Number of residual blocks in this layer group.
        stride (int): Stride for the first block in the group.

        Returns:
        nn.Sequential: Sequential container of residual blocks for this layer group.
        """

        # First block may downsample, others maintain stride of 1
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            # Update the input channels for the next block
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the ResNet-34 model.

        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)

        # Flatten the spatial dimensions for the fully connected layer
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# Instantiate the model.
model = ResNet34(ResidualBlock, [3, 4, 6, 3]).to(device)
model.to(device)
# # Load the state_dict into the model
model_path = "resnet34_cifar10_checkpoint.pth"
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])


# Evaluate the model on the test set
accuracy, precision, recall, f1, cm = evaluate(test_loader, model, device)

# Print the evaluation metrics
print("\nModel Evaluation:")
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot the confusion matrix
class_names = test_set.classes
plot_confusion_matrix(cm, class_names)