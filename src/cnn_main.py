# Import the necessary modules
import os
# Set the environment variable to avoid OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.utils.data
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms import RandomRotation, RandomHorizontalFlip, ColorJitter, RandomGrayscale
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
# Initialize the tensorboard writer
writer = SummaryWriter('logs/run/fahadtest2')

# Check if CUDA is available and set the device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Use the first GPU device
else:
    device = torch.device("cpu")  # Use CPU if CUDA is not available

# Set the CUDA device
torch.cuda.set_device(device)

# Define the path to the extracted data
extracted_dir_path = 'C:\\ML_exercise\\ml_new'
# Load the labels from a text file
labels = np.loadtxt(os.path.join(extracted_dir_path, "label_train.txt"))
# Define the paths to the image files
image_paths = [os.path.join(extracted_dir_path, "train_img", f"{i+1:06d}.jpg") for i in range(len(labels))]

# Split the data into training and validation sets
train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

# Define the transformation to be applied on the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the dataset class
class ImageClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment

        # Define transformations for augmentation
        self.augmentations = transforms.Compose([
            RandomRotation(30),
            RandomHorizontalFlip(),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            RandomGrayscale(p=0.1),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        if self.augment:
            img = self.augmentations(img)

        label = self.labels[idx]
        
        return img, label

# Identify the minority class
unique, counts = np.unique(train_labels, return_counts=True)
minority_class = unique[np.argmin(counts)]

# Apply augmentations only to the minority class
train_image_paths_augmented = []
train_labels_augmented = []

for img_path, label in zip(train_image_paths, train_labels):
    train_image_paths_augmented.append(img_path)
    train_labels_augmented.append(label)

    if label == minority_class:
        for _ in range(7):  # Augment the image 7 times
            train_image_paths_augmented.append(img_path)
            train_labels_augmented.append(label)

# Create the augmented train dataset
train_dataset = ImageClassificationDataset(
    image_paths=train_image_paths_augmented,
    labels=train_labels_augmented,
    transform=transform,
    augment=True  # Apply augmentations
)

val_dataset = ImageClassificationDataset(
    image_paths=val_image_paths,
    labels=val_labels,
    transform=transform
)
# Define the batch size
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Create the data loaders
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.fc1(x)
        return x

# Instantiate the model
net = Net()
net.to(device)
# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

num_epochs = 500
patience = 30
# Define the number of epochs and patience for early stopping
best_hter = np.inf
counter = 0

# Print the number of training and validation images
print(f"Number of training images: {len(train_dataset)}")
print(f"Number of validation images: {len(val_dataset)}")

# Compute and print the class count
class_count_train = np.bincount(train_labels.astype(int))
class_count_val = np.bincount(val_labels.astype(int))

print(f"Training class count: {class_count_train}")
print(f"Validation class count: {class_count_val}")

# Define the weight for computing the weighted HTER
weight = 0.2

# Initialize lists to store the losses and accuracies for each epoch
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Loop over the epochs
for epoch in range(num_epochs):
    # Training
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(train_loader, 0):
        # Get the inputs and labels
        inputs, labels = data[0].to(device), data[1].to(device)

        # Convert labels to LongTensor
        labels = labels.long()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)
        
        # Compute the accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Backward pass
        loss.backward()

        # Optimize
        optimizer.step()

        # Print statistics
        running_loss += loss.item()

    train_accuracy = correct / total
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(train_accuracy)
    print('Epoch %d, training loss: %.3f, training accuracy: %.3f' % (epoch + 1, running_loss / len(train_loader), train_accuracy))
    
    # Evaluation
    net.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    with torch.no_grad():
        # Compute the predictions for the validation set and the confusion matrix components
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            labels = labels.long()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            TP += ((predicted == 1) & (labels == 1)).sum().item()
            FP += ((predicted == 1) & (labels == 0)).sum().item()
            TN += ((predicted == 0) & (labels == 0)).sum().item()
            FN += ((predicted == 0) & (labels == 1)).sum().item()

            # Compute the loss
            loss = criterion(outputs, labels)
            
            # Print statistics
            running_loss += loss.item()

        val_accuracy = correct / total
        val_losses.append(running_loss / len(val_loader))
        val_accuracies.append(val_accuracy)
        print('Epoch %d, validation loss: %.3f, validation accuracy: %.3f' % (epoch + 1, running_loss / len(val_loader), val_accuracy))

        # Compute the weighted HTER
        FAR = FP / (FP + TN)
        FRR = FN / (FN + TP)
        HTER = weight * FRR + (1 - weight) * FAR

        # Print statistics
        print('Epoch %d, HTER: %.3f' % (epoch + 1, HTER))

        # Check if this is the best model
        if HTER < best_hter:
            best_hter = HTER
            # Save the model parameters
            torch.save(net.state_dict(), 'C:\\ML_exercise\\ml_new\\models\\best_model2.pth')
            print(f'Epoch {epoch + 1}, best model saved with HTER: {best_hter}')
            counter = 0
        else:
            counter += 1

        # Early stopping
        if counter >= patience:
            print('Early stopping')
            break

print('Finished Training')

# Plot the losses and accuracies
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(len(train_losses)), train_losses, label='Training loss')
plt.plot(range(len(val_losses)), val_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()
plt.subplot(1, 2, 2)
plt.plot(range(len(train_accuracies)), train_accuracies, label='Training accuracy')
plt.plot(range(len(val_accuracies)), val_accuracies, label='Validation accuracy')
plt.legend(frameon=False)
plt.show()






