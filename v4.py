import numpy as np
from balance import *
#from torchsummary import summary
import torch
torch.cuda.empty_cache()

print(f'PyTorch version: {torch.__version__}')
print("GPU found :)" if torch.cuda.is_available() else "No GPU :(")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
multi_image_dataset = torch.load('datasets/multi_image_dataset.pt')
nb_images = 50

#sizes = list(map(lambda x: x['image'].shape, multi_image_dataset))
#print(set(sizes))
class ModelV4(nn.Module):
    
    def __init__(self, input_channels=3*nb_images):
        """Convnet with 4 convolution layer + pooling + BN, with 3 fully connected at the end"""
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=256, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, 3)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 256, 3)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 64, 3)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(64*14*14 , 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(0.5)
        
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 64*14*14) # Flatten
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        # Binary classification
        out = self.sigmoid(self.fc3(x))
        
        return out

model = ModelV4().to(device)
#summary(model, (150, 256, 256))
batch_size = 1
ratio=0.2

train_loader, test_loader = train_test_split(multi_image_dataset, ratio, batch_size)
num_epoch = 10
step_count = len(train_loader)
loss_function = nn.BCELoss()
losses = list()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epoch):
    
    epoch_loss = 0
    
    for i, sample in enumerate(train_loader):
        
        image = sample['image'].to(device, dtype=torch.float)
        target = sample['target'].to(device, dtype=torch.float)
        target = target.view(target.shape[0], 1)
        
        # Reset gradiant
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(image)
        
        # Compute loss
        loss = loss_function(pred, target)
        epoch_loss += loss
        
        # Backprop
        loss.backward()
        optimizer.step()
        
        del image
        del target
        del pred
        del sample
        
        # Debug
        if((i+1) % 10 == 0):
            print(
                        f"Epoch [{epoch + 1}/{num_epoch}]"
                        f", step [{i + 1}/{step_count}]"
                        f", loss: {loss.item():.4f}"
                    )
            
    losses.append(epoch_loss)

# Save model 
torch.save(model.state_dict(), './models/model_v4.h5')
del model