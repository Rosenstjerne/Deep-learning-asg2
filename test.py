import os
import torch
import numpy as np
from train import PneumoniaClassifier
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
# Author Mikkel Asmussen
def txt_to_img(file_path):
    with open(file_path) as f:
        pixels = f.read().split()
    
    pixels = np.asarray(pixels)
    pixels = pixels.astype(float).astype(int)
    pixels = pixels.reshape(224, 224)

    return Image.fromarray(np.uint8(pixels), 'L')

def img_to_tensor(image):
    img_as_arr = np.float32(image)
    img_as_tensor = torch.from_numpy(img_as_arr).float()
    return img_as_tensor.view(1, 224, 224)


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.files = []
        self.label_dict = {
            Path('data/data/test/NORMAL'): 0,
            Path('data/data/test/PNEUMONIA'): 1
        }

        for (root, _, filenames) in os.walk(self.root_dir):
            for name in filenames:
                self.files.append(Path(os.path.join(root, name)))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imagepath = self.files[idx]
        image = img_to_tensor(txt_to_img(imagepath))
        label = self.label_dict[imagepath.parent]

        if self.transform:
            image = self.transform(image)

        return image, label
        
def eval_model(model, test_loader):
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()
    correct, total = 0, 0

    print('--- TEST BEGIN ---')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    print(f'Correct evaluations: {correct}/{total}')
    print(f'Test accuracy: {correct / total}')
    print('--- TEST END ---')

if __name__ == '__main__':
    model = torch.load('model.pth')

    test_dataset = TestDataset('./data/data/test')
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    eval_model(model, test_loader)
