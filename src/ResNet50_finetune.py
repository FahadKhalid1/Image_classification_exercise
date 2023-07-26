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
writer = SummaryWriter('logs/run/fahadtest')


if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Use the first GPU device
else:
    device = torch.device("cpu")  # Use CPU if CUDA is not available

torch.cuda.set_device(device)

# Define the transforms to apply to the images
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
])

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, labels_file, transform=None):
        self.image_folder = image_folder
        self.labels_file = labels_file
        self.transform = transform
        self.initial_image_paths = self.load_image_paths()
        self.image_paths, self.labels = self.filter_images_with_equal_labels()

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


def hter_loss(outputs, targets):
    targets = targets.view(-1, 1)  # Change the shape of targets to [batch_size, 1]
    probabilities = torch.sigmoid(outputs)
    predictions = torch.round(probabilities)
    errors = torch.abs(predictions - targets)
    false_acceptance = torch.mean(errors * (1 - targets))
    false_rejection = torch.mean(errors * targets)
    half_total_error = (false_acceptance + false_rejection) / 2.0
    return half_total_error

# Define the custom dataset and dataloader
root_folder = 'C:\ML_exercise\ml_exercise_therapanacea'
dataset = CustomDataset(os.path.join(root_folder, 'train_img'), os.path.join(root_folder, 'label_train.txt'), transform=transform)

# get dataset distribution

# Split the dataset into train and validation sets
train_size = int(0.8 * len(dataset))  # 80% for training, 20% for validation
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Define the dataloaders for training and validation
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

# # Function to plot images with labels
def plot_images(image_paths, images, labels):
    fig, axes = plt.subplots(4, 2, figsize=(12, 6))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        image = images[i]
        label = labels[i]

        print(f'image name: {image_paths[i]} and label: {label}')

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

# loading a pretrained ResNet50 model for image classification and customizes it for a 
# specific task. It replaces the model's final fully connected layer with a new sequence 
# of layers to adapt to a new classification problem. After that, it freezes all layers 
# except for the last block (Stage 4) to preserve prelearned features and only fine-tune 
# the last block. This technique is called transfer learning. Finally, the customized 
# model is transferred to the computing device for further use.

model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
num_features = model.fc.in_features

# Create additional layers for your custom classification task
additional_layers = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Linear(256, 2),
)

# Combine the pre-trained model and additional layers
model.fc = additional_layers

# Freeze all layers except for Stage 4
for name, param in model.named_parameters():
    if not name.startswith('layer4'):
        param.requires_grad = False
    else:
        param.requires_grad = True

model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
loss_fn = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)

# Train
num_epochs = 500
log_interval = 10
global_step = 0

best_val_loss = float('inf')
best_model_state = None

early_stopping_limit = 50
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
        model_name = 'R50_finetune.pt'
        bestmodel_folder = r'C:\ML_exercise\ml_exercise_therapanacea\models'
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

