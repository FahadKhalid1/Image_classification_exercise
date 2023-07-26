import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.utils.data
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import torch.nn.functional as F
from torchvision.io import read_image
import torchvision.models as models

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformations for the input images
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = os.listdir(root_dir)
        
    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_path = os.path.join(self.root_dir, image_file)
        image = read_image(image_path).float()
        image = image.unsqueeze(0)
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image_file, image

    def __len__(self):
        return len(self.image_files)

def main():
    # Initialize the model
    model = models.resnet50(pretrained=False)  # Set pretrained=False because we will load our own weights

    # Create additional layers for your custom classification task
    additional_layers = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Linear(256, 2),
    )

    # Replace the fully connected layer of the model
    model.fc = additional_layers

    # Freeze all layers except for Stage 4
    for name, param in model.named_parameters():
        if not name.startswith('layer4'):
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Load the state dict to the model
    model.load_state_dict(torch.load('models/R50_finetune.pt'))

    # Send the model to the device
    model = model.to(device)
    model.eval()



    # # Transformations for the input images
    # transform = transforms.Compose([
    #     transforms.Resize((64, 64)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])

    val_img_folder = UnlabeledDataset(r'C:\ML_exercise\ml_exercise_therapanacea\val_img', transform=transform)
    val_dataloader = torch.utils.data.DataLoader(val_img_folder, batch_size=1, shuffle=False, num_workers=4)


    # DataFrame to hold results
    results = []

    # Predict on the validation dataset
    for image_file, inputs in val_dataloader:  # Now you get image_file directly from the DataLoader
        inputs = inputs.squeeze(0).to(device)

        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        # Detach predictions
        preds = preds.detach().cpu().numpy()

        results.append([image_file[0], preds[0]])  # image_file is a 1-element batch, so get the first (and only) element

    # Create DataFrame and save to Excel
    df = pd.DataFrame(results, columns=['Image', 'Prediction'])
    df.to_excel('predictions.xlsx', index=False)

if __name__ == '__main__':
    main()
