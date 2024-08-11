from torch.utils.data import DataLoader

from carvana_dataset import val_loader, train_loader
from model import UNet
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import load_checkpoint, save_checkpoint

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOAD_MODEL = False
NUM_EPOCHS = 10
model = UNet().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

def train_model(loader: DataLoader):
    loop = tqdm(loader)

    for idx, (x_data, y_masks) in enumerate(loop):

        images = x_data.to(device=DEVICE)
        # unsqueeze 1 - dimension
        masks = y_masks.float().unsqueeze(1).to(device=DEVICE)

        with torch.cuda.amp.autocast():
            predictions = model(images)
            loss = criterion(predictions, masks)

        # reset gradients
        optimizer.zero_grad()
        # update weights and biases, compute gradients
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

def check_accuracy(loader, model, device=DEVICE):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def main():
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    check_accuracy(val_loader, model, device=DEVICE)

    for epoch in range(NUM_EPOCHS):
        train_model(train_loader)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

if __name__ == "__main__":
    main()
