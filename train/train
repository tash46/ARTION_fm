import sys
import os
import torch
import argparse
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from models.x3 import X3   #import models
from datasets import get_loaders_from_config

def multiclass_dice_score(preds, targets, num_classes, ignore_index=None):
    """
    preds: [B, C, H, W] (raw logits from model)
    targets: [B, H, W] (ground truth class labels)
    """
    preds = torch.argmax(preds, dim=1)  # [B, H, W]

    dice_scores = []
    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            continue

        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        dice_scores.append(dice.item())

    return sum(dice_scores) / len(dice_scores)

class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1.0, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, preds, targets):
        """
        preds: raw logits of shape [B, C, H, W]
        targets: ground truth of shape [B, H, W] with class indices
        """
        preds = F.softmax(preds, dim=1)  # Convert to probabilities [B, C, H, W]

        dice_loss = 0.0
        for cls in range(self.num_classes):
            if self.ignore_index is not None and cls == self.ignore_index:
                continue

            pred_flat = preds[:, cls].contiguous().view(-1)
            target_flat = (targets == cls).float().contiguous().view(-1)

            intersection = (pred_flat * target_flat).sum()
            union = pred_flat.sum() + target_flat.sum()
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss += 1 - dice  # Dice Loss is 1 - Dice Coefficient

        return dice_loss / self.num_classes

# Combined loss
class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, num_classes=4):
        super(CrossEntropyDiceLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes)

    def forward(self, preds, targets):
        ce_loss = self.ce(preds, targets)
        dice_loss = self.dice(preds, targets)
        return ce_loss + dice_loss

def train_x3(labeled_loader, val_loader, num_epochs=10):    #change the model name here
    print("Start Training.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = X3(num_classes=4).cuda()    #change the model here
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    criterion = CrossEntropyDiceLoss(num_classes=4).to(device)
    best_val_dice = 0.0
    best_model_path = f'D:/'  #best model saving path

    train_losses = []
    val_losses = []
    val_dices = []

    log_file_path = f'D:/'  #training log file saving path
    with open(log_file_path, 'w') as log_file:
        log_file.write("Epoch\tTrain Loss\tVal Loss\tVal Dice\n")

        for epoch in range(num_epochs):
            start_time = time.time()
            model.train()
            train_loss = 0
            print(f"Starting epoch {epoch + 1}/{num_epochs}")
            for i, batch in enumerate(labeled_loader):
                if len(batch) == 2:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                else:
                    print(f"Unexpected batch format : {batch}")
                    continue

            val_loss = 0
            val_dice = 0
            model.eval()
            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                    val_outputs = model(val_inputs)
                    val_loss += criterion(val_outputs, val_targets).item()

                    val_dice += multiclass_dice_score(val_outputs, val_targets, num_classes=4)

            train_loss /= len(labeled_loader)
            val_loss /= len(val_loader)
            val_dice /= len(val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_dices.append(val_dice)

            epoch_duration = time.time() - start_time
            print(f"Epoch {epoch + 1}/{num_epochs}, Duration: {epoch_duration:.2f}s, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

            # Log the metrics to the file
            log_file.write(f"{epoch + 1}\t{train_loss:.4f}\t{val_loss:.4f}\t{val_dice:.4f}\n")

            # Save the model with the best validation dice
            if val_dice > best_val_dice: 
                best_val_dice = val_dice
                torch.save(model.state_dict(), best_model_path)
                print(f"Model saved with val dice: {val_dice:.4f}")

    # Plotting
    epochs = range(1, num_epochs + 1)
    fig, ax1 = plt.subplots()  
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(epochs, train_losses, label='Train Loss', color='tab:blue')
    ax1.plot(epochs, val_losses, label='Val Loss', color='tab:cyan')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Dice Score', color='tab:red')
    ax2.plot(epochs, val_dices, label='Val Dice', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title('Training and Validation Metrics')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.savefig(f'./metrics.png')  #save metrics plot
    
    #return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Path to the config file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    args = parser.parse_args()

    train_loader, val_loader = get_loaders_from_config(args.config, batch_size=args.batch_size)
    
    train_x3(train_loader, val_loader, num_epochs=args.epochs)    #change the function name here
    
