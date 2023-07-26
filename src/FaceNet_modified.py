# Importing necessary libraries and modules
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.utils.data
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, RandomVerticalFlip, RandomCrop, ColorJitter, RandomAffine, RandomErasing
from collections import Counter
from sklearn.utils import resample
import numpy as np

# Creating a TensorBoard writer
writer = SummaryWriter('logs/run/fahad5')

# Set the device to GPU if available, else CPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Use the first GPU device
else:
    device = torch.device("cpu")  # Use CPU if CUDA is not available

# Set the current device
torch.cuda.set_device(device)

# Define the transforms to apply to the images
# Here, we are only converting the image to tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Augmentation transforms
# These will be applied only to the minority class to balance the dataset
augment = transforms.Compose([
    RandomHorizontalFlip(p=0.2),
    # RandomVerticalFlip(p=0.5),
    RandomRotation(10),
    # RandomCrop(224, padding=4),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    RandomAffine(degrees=0, translate=(0.1, 0.1)),
    # RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    transforms.ToTensor(),
])

# Custom Dataset class
class CustomDataset(torch.utils.data.Dataset):
    # Initialization method for the dataset
    def __init__(self, image_folder, labels_file, transform=None, augment=None):
        self.image_folder = image_folder
        self.labels_file = labels_file
        self.transform = transform
        self.augment = augment
        self.image_paths = self.load_image_paths()
        self.labels = self.load_labels()
        self.balance_classes()  # Balance the dataset

    # Method to load image paths
    def load_image_paths(self):
        image_paths = []
        # Iterate over each file in the image_folder
        for filename in sorted(os.listdir(self.image_folder)):
            # Check if the file is an image
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                # Append the full image path to the list
                image_paths.append(os.path.join(self.image_folder, filename))
        return image_paths

    # Method to load labels from the labels file
    def load_labels(self):
        labels = []
        with open(self.labels_file, 'r') as file:
            for line in file:
                # Convert the label from string to integer and append to the list
                label = int(line.strip())
                labels.append(label)
        return labels

    # Method to balance the classes
    # The balance_classes function is designed to handle class imbalance in a dataset by ensuring
    # an equal number of samples from both classes. It identifies the class with fewer instances 
    # (minority class), and duplicates its instances until it matches the count of the majority class.
    # This upsampled dataset is then shuffled to ensure a good mix of classes. The function updates
    # the labels and image paths accordingly, providing a balanced dataset for model training.

    def balance_classes(self):
        # Get the indices for each class
        class_zero = [i for i, label in enumerate(self.labels) if label == 0]
        class_one = [i for i, label in enumerate(self.labels) if label == 1]

        # Get the minority class
        minority_class = class_zero if len(class_zero) < len(class_one) else class_one

        # Resample the minority class to have the same number of samples as the majority class
        minority_class_upsampled = resample(minority_class, replace=True, 
                                            n_samples=len(class_one), random_state=123)

        # Combine the majority class with the upsampled minority class
        balanced_indices = np.concatenate([class_one, minority_class_upsampled])

        # Shuffle the combined dataset
        np.random.shuffle(balanced_indices)

        # Use the shuffled indices to create the balanced dataset
        self.labels = [self.labels[i] for i in balanced_indices]
        self.image_paths = [self.image_paths[i] for i in balanced_indices]

    # Method to get the length of the dataset
    def __len__(self):
        return len(self.image_paths)

    # Method to get an item from the dataset at a particular index
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # Open the image and convert it to RGB
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        # If the image belongs to the minority class, apply augmentation, else apply the regular transform
        if label == 0 and self.augment:
            image = self.augment(image)
        elif self.transform:
            image = self.transform(image)

        return image_path, image, label

# Define the root folder where the images and labels are stored
root_folder = 'C:/ML_exercise/ml_new'

# Create an instance of the CustomDataset
dataset = CustomDataset(os.path.join(root_folder, 'train_img'), os.path.join(root_folder, 'label_train.txt'), transform=transform, augment=augment)

# Create a DataLoader from the CustomDataset
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

def hter_loss(outputs, targets):
    targets = targets.view(-1, 1)  # Change the shape of targets to [batch_size, 1]
    probabilities = torch.sigmoid(outputs)
    predictions = torch.round(probabilities)
    errors = torch.abs(predictions - targets)
    false_acceptance = torch.mean(errors * (1 - targets))
    false_rejection = torch.mean(errors * targets)
    half_total_error = (false_acceptance + false_rejection) / 2.0
    return half_total_error


# get dataset distribution

# Split the dataset into train and validation sets
train_size = int(0.8 * len(dataset))  # 80% for training, 20% for validation
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Count the number of examples in each class in the training set
train_label_counts = {0: 0, 1: 0}
for _, _, label in train_dataset:
    train_label_counts[label] += 1

# Count the number of examples in each class in the validation set
val_label_counts = {0: 0, 1: 0}
for _, _, label in val_dataset:
    val_label_counts[label] += 1

# Define the dataloaders for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Print the total number of examples and the count for each class
print(f"Total number of training examples: {len(train_dataset)}")
print(f"Number of examples in class 0 (train): {train_label_counts[0]}")
print(f"Number of examples in class 1 (train): {train_label_counts[1]}")
print(f"Total number of validation examples: {len(val_dataset)}")
print(f"Number of examples in class 0 (validation): {val_label_counts[0]}")
print(f"Number of examples in class 1 (validation): {val_label_counts[1]}")


# # Function to plot images with labels
def plot_images(image_paths, images, labels):
    fig, axes = plt.subplots(8, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        image = images[i]
        label = labels[i]

        print(f'image name: {image_paths[i]} and label: {label}')
        
        # Unnormalize the image
        #image = (image * 0.5) + 0.5
        
        # Convert the image tensor to numpy array and transpose the dimensions
        image = image.permute(1, 2, 0).numpy()
        
        # Plot the image and print the label on it
        ax.imshow(image)
        ax.set_title(str(label))
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage:
# Load a batch of images and labels from the train dataloader
image_paths, images, labels = next(iter(train_dataloader))

# Plot the images with labels
plot_images(image_paths, images, labels)

# This is a simplified variant of the FaceNet model, built using PyTorch. The model's architecture
# includes several convolutional layers for feature extraction, batch normalization layers for 
# input standardization, ReLU activation functions for introducing non-linearity, max-pooling
# layers for downsampling, and fully connected layers for classification. The model also utilizes 
# dropout for regularization. The forward method outlines how these layers interact during the 
# model's forward pass. Finally, an instance of the model is created and moved to the GPU if one is available.

class FaceNetModel(nn.Module):
    def __init__(self, embedding_dimension=128, num_classes=2):
        super(FaceNetModel, self).__init__()
        
        # Define your own architecture based on FaceNet
        # Here is a simplified version
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(192)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(192, embedding_dimension)
        self.fc2 = nn.Linear(embedding_dimension, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x
    

model = FaceNetModel().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)



# Train
num_epochs = 500
log_interval = 10
global_step = 0

best_val_loss = float('inf')
best_model_state = None

early_stopping_limit = 55
early_stopping_counter = 0

train_loss_list = []
val_loss_list = []


for epoch in range(num_epochs):
    model.train()
    commulative_train_loss = 0
    log_interval_train_loss = 0

    for batch_idx, (_, images, labels) in enumerate(train_dataloader):
        images = images.float().to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = hter_loss(outputs, labels)  # Change this line to use the hter_loss
        commulative_train_loss += loss.item()
        log_interval_train_loss += loss.item()
        loss.backward()
        optimizer.step()

        # Increment global step counter
        global_step += 1

        # Print training progress
        if batch_idx % log_interval == 0 and not batch_idx == 0:
            writer.add_scalar('Loss/train', loss.item(), global_step)
            print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_dataloader)} | Interval Train Loss: {log_interval_train_loss/log_interval:.4f}")
            log_interval_train_loss = 0

    train_loss_list.append(commulative_train_loss / len(train_dataloader))  # Append training loss

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for _, images, labels in val_dataloader:
            images = images.float().to(device)
            labels = labels.to(device)
            outputs = model(images)
            val_loss += hter_loss(outputs, labels).item()  # Change this line to use the hter_loss

    val_loss /= len(val_dataloader)
    val_loss_list.append(val_loss)  # Append validation loss        

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0

        # Save the model in the specified directory
        best_model_state = model.state_dict()
        model_name = 'Face_aug_bestmodel.pt'
        bestmodel_folder = r'C:\ML_exercise\ml_new\models'
        model_path = os.path.join(bestmodel_folder, model_name)
        torch.save(best_model_state, model_path)
        print(f"Val_loss decreased from {best_val_loss / len(val_dataloader):.4f} to {val_loss / len(val_dataloader):.4f}. Saving the model to {model_path}.")
    else:
        epochs_no_improve += 1
        if epochs_no_improve == early_stopping_limit:
            print(f'Early stopping! Best validation loss: {best_val_loss}')
            break  # exit loop

    # Log validation loss and accuracy to TensorBoard
    val_loss /= len(val_dataloader)
    writer.add_scalar('Loss/val', val_loss, epoch)

plt.figure(figsize=(10, 5))
plt.title("Training and Validation Loss")
plt.plot(train_loss_list,label="Training")
plt.plot(val_loss_list,label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# Close TensorBoard writer
writer.close()
