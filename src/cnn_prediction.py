import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
import torchvision.models as models
from torchvision.models import resnet50
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F


# Specify the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
model = Net()

model.load_state_dict(torch.load('C:\\ML_exercise\\ml_new\\models\\best_model2.pth'))
model.to(device)
model.eval()  # Set the model to evaluation mode

# Specify the transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Specify the path to the new images
img_dir = 'C:\\ML_exercise\\ml_new\\val_img'

# Prepare a list to store the image names and predictions
predictions = []

# Loop over all images
for img_name in tqdm(os.listdir(img_dir), desc="Predicting labels"):
    img_path = os.path.join(img_dir, img_name)
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    
    # Make a prediction
    output = model(img)
    _, predicted = torch.max(output, 1)
    
    # Store the image name and prediction
    predictions.append([img_name, predicted.item()])

# Convert the list to a pandas DataFrame
df = pd.DataFrame(predictions, columns=['Image', 'Label'])

# Save the DataFrame to an Excel file
df.to_excel('C:\\ML_exercise\\ml_new\\NewExp\\new_Pred_sam.xlsx', index=False)