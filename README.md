# ResNet-34 CIFAR-10 Evaluation - Colab Setup Guide

This guide explains how to set up and execute the `demo.py` script in a Google Colab environment to evaluate the ResNet-34 model on the CIFAR-10 dataset.

---

## Requirements

1. **Google Colab** account.
2. **Python 3.8+** with the following libraries installed:
   - `torch`
   - `torchvision`
   - `numpy`
   - `scikit-learn`
   - `matplotlib`
   - `seaborn`
3. **Pre-trained Model**: 
   - `resnet34_cifar10_checkpoint.pth` containing the trained model's `state_dict`.

---

## Steps to Run `demo.py`

### Step 1: Upload Necessary Files

1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new notebook.
3. Connect to a GPU runtime.
4. Upload the following files:
   - `demo.py`
   - `resnet34_cifar10_checkpoint.pth` - https://drive.google.com/drive/folders/1KC69I7xvrrNorvFtKP_r9HiQyLz3-Xxb

### Step 2: Install Required Libraries

Run the following code in a Colab cell to install all necessary dependencies:

```python
!pip install torch torchvision matplotlib seaborn scikit-learn
```

### Step 3: Prepare the Environment

Ensure that the CIFAR-10 dataset is downloaded, and the script can access the GPU if available. Add the following setup code:

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### Step 5: Run the Script

Use the following code snippet to run the ```demo.py``` script:

```python
!python demo.py
```

### Step 6: View Results (Confusion Matrix)

After execution:
- The evaluation metrics will be displayed in the Colab output.
- A confusion matrix plot (confusion_matrix.jpeg) will be saved in the current working directory. Use the following code to download it:
```python
from google.colab import files
files.download('confusion_matrix.jpeg')
```

