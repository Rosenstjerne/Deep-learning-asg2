import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

from layers import CustomLinear, CustomConv2d

class PneumoniaClassifier(nn.Model):
    def __init__(self):
        super(PneumoniaClassifier, self).__init__()

    def forward(self, x):
        return x

def train_step(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss, correct, total = 0, 0, 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)  # move data to GPU
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)  # compute loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.size(0)  # accumulate the loss
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    avg_loss = train_loss / len(train_loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy

def validate_step(model, validate_loader, criterion, device):
    model.eval()
    validate_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for data, target in validate_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            validate_loss += loss.item() * data.size(0)  # accumulate the loss
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = validate_loss / len(validate_loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy

def train_and_validate(model, train_loader, validate_loader, optimizer, criterion, device, epochs):
    results = {
        "train_loss": [],
        "train_acc": [],
        "validate_loss": [],
        "validate_acc": []
    }

    scheduler = ReduceLROnPlateau(optimizer, 'min')

    for epoch in range(epochs):
        train_loss, train_acc = train_step(model, train_loader, criterion, optimizer, device)
        validate_loss, validate_acc = validate_step(model, validate_loader, criterion, device)

        scheduler.step(validate_loss)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["validate_loss"].append(validate_loss)
        results["validate_acc"].append(validate_acc)

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Validate Loss: {validate_loss:.4f}, Validate Acc: {validate_acc:.4f}')

    return results

if __name__ == '__main__':
    train_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        ])
        
    train_dataset = datasets.ImageFolder('./data/data/training', train_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = PneumoniaClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda")
    model = model.to(device)
    print(device)

    epochs = 10
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model, train_loader, criterion, optimizer, device)

        print(f'Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}')

    torch.save(model, './pneumonia_classifier.pth')
