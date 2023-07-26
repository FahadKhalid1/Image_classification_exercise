# Import necessary libraries
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Overcomes potential duplication of dynamic libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter # TensorBoard for visualizing training progress
from torchvision.models import resnet50
from torchinfo import summary  # Provides a detailed model summary
from timm.models import create_model # timm is a library for image models
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import matplotlib
matplotlib.use('TkAgg')  # Specifies matplotlib backend
import matplotlib.pyplot as plt
writer = SummaryWriter('logs/run/fahad5')  # Specifies the log directory for TensorBoard


# Check for CUDA availability for GPU utilization and set the device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Use the first GPU device
else:
    device = torch.device("cpu")  # Use CPU if CUDA is not available

torch.cuda.set_device(device)


# Specify transformations to be applied on images
transform = transforms.Compose([
    transforms.ToTensor(),
])


# Define custom dataset class for handling our specific image dataset
class CustomDataset(torch.utils.data.Dataset):
    # It loads images, pairs them with their corresponding labels, and applies transformations
    def __init__(self, image_folder, labels_file, transform=None):
        self.image_folder = image_folder
        self.labels_file = labels_file
        self.transform = transform
        self.initial_image_paths = self.load_image_paths()
        self.image_paths, self.labels = self.filter_images_with_equal_labels()

        # Calculate class distribution
        self.class_counts = {0: self.labels.count(0), 1: self.labels.count(1)}

    def load_image_paths(self):
        image_paths = []
        for filename in sorted(os.listdir(self.image_folder)):
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                image_paths.append(os.path.join(self.image_folder, filename))
        return image_paths
################################################# Class Balancing #####################################################
     # The function filter_images_with_equal_labels is part of a custom dataset class and is 
     # used to balance the dataset. Initially, it reads a file containing image labels, counting
     # the occurrence of each label assumed to be either 0 or 1. The function then establishes
     #  the smaller count between the two labels to balance the dataset. Subsequently, it iterates
     #  over the label file again, adding the corresponding image path and label to their 
     # respective lists and incrementing the label count, but only if the current label 
     # count is less than or equal to this minimum count. This ensures an equal number 
     # of images for each label, creating a balanced dataset. The function returns these
     #  lists of selected image paths and their corresponding labels.
    def filter_images_with_equal_labels(self):
        image_paths = []
        labels = []
        label_counts = {0: 0, 1: 0}

        # Count the number of occurrences for each label
        with open(self.labels_file, 'r') as file:
            for line in file:
                label = int(line.strip())
                label_counts[label] += 1

        # Determine the minimum count for labels 0 and 1
        min_count = min(label_counts[0], label_counts[1])
        
        label_counts = {0: 0, 1: 0}

        # Filter the images based on the minimum count
        with open(self.labels_file, 'r') as file:
            for idx, line in enumerate(file):
                image_path = self.initial_image_paths[idx]
                # image_path = os.path.join(self.image_folder, line.strip() + '.jpg')
                label = int(line.strip())
                if label_counts[label] <= min_count:
                    image_paths.append(image_path)
                    labels.append(label)
                    label_counts[label] += 1

        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image_path, image, label

    def get_class_counts(self):
        counts = {0: 0, 1: 0}
        for label in self.labels:
            counts[label] += 1
        return counts


# Define a custom loss function (half total error rate loss)
def hter_loss(outputs, targets):
    targets = targets.view(-1, 1)  # Change the shape of targets to [batch_size, 1]
    probabilities = torch.sigmoid(outputs)
    predictions = torch.round(probabilities)
    errors = torch.abs(predictions - targets)
    false_acceptance = torch.mean(errors * (1 - targets))
    false_rejection = torch.mean(errors * targets)
    half_total_error = (false_acceptance + false_rejection) / 2.0
    return half_total_error

# Define the path to the dataset and create the dataset object
root_folder = 'C:\ML_exercise\ml_new'
dataset = CustomDataset(os.path.join(root_folder, 'train_img'), os.path.join(root_folder, 'label_train.txt'), transform=None)

# Split the dataset into train and validation sets
train_size = int(0.8 * len(dataset))  # 80% for training, 20% for validation
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Set the data configuration using the timm library
config = resolve_data_config({}, model=create_model('vit_base_patch16_224'))
train_transform = create_transform(**config)
val_transform = create_transform(**config)

# Apply the transforms to the datasets
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

# Create DataLoader for efficient loading and batch processing of data during training
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)


print(f"Images in training set: {len(train_dataloader.dataset)}")
print(f"Images in validation set: {len(val_dataloader.dataset)}")

# Count the class labels in training and validation datasets
train_labels = [label for _, _, label in train_dataset]
val_labels = [label for _, _, label in val_dataset]
train_counts = {0: train_labels.count(0), 1: train_labels.count(1)}
val_counts = {0: val_labels.count(0), 1: val_labels.count(1)}

print(f"Class count in training set: {train_counts}")
print(f"Class count in validation set: {val_counts}")

# Use the timm library to create a pretrained vision transformer model
# Create an instance of the pre-trained 'vit_base_patch16_224' model
# 'vit_base_patch16_224' is a Vision Transformer model with a base configuration, using 16x16 patches
# The model is pre-trained on a large dataset, likely ImageNet
# The number of output classes is set to 2, which suggests a binary classification task
model = create_model('vit_base_patch16_224', pretrained=True, num_classes=2).to(device)

# For the parameters in the head of the model, we enable gradients.
# The 'head' is typically the final fully connected layer in the model which maps the extracted
# features to the output classes.
# By enabling gradients here, we are allowing these parameters to be updated during training.
# This is a common strategy for transfer learning: the base model is frozen and used as a fixed
# feature extractor, while the final classification layer is trained on the new data.

for param in model.parameters():
    param.requires_grad = False

for param in model.head.parameters():
    param.requires_grad = True

# Print a summary of the model. This includes the number and types of layers in the model,
# as well as the number of parameters.
# The 'input_size' argument specifies the size of the input that the model expects.
# In this case, the model expects input of size (64, 3, 224, 224) which represents a
# mini-batch of 64 images, each of size 224x224 with 3 color channels (RGB).
summary(model, input_size=(64, 3, 224, 224))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)

writer = SummaryWriter('logs/run/fahad5')
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
        model_name = 'vit_bestmodel.pt'
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

# Plot the Training and validation loss curevs
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
