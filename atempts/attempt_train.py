import torch.optim as optim
import torch
import torch.nn as nn
from carvana_dataset import train_loader
from model import UNet

# Initialize the model, loss function, and optimizer
model = UNet()
criterion = nn.CrossEntropyLoss()  # Suitable for multi-class segmentation
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # Number of epochs
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        # Move images and labels to GPU if available
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        else:
            print("CUDA is not available. Running on CPU...")
            images = images
            labels = labels

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels.long())  # Convert labels to long for CrossEntropyLoss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/10], Loss: {running_loss / len(train_loader):.4f}')

print('Training complete.')
