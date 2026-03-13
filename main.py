from src.dataset_loader import LEVIRDataset
from torch.utils.data import DataLoader

train_path = "dataset/LEVIR CD/train"
val_path = "dataset/LEVIR CD/val"
test_path = "dataset/LEVIR CD/test"

train_dataset = LEVIRDataset(train_path)
val_dataset = LEVIRDataset(val_path)
test_dataset = LEVIRDataset(test_path)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

for imgA, imgB, label in train_loader:

    print("Image A shape:", imgA.shape)
    print("Image B shape:", imgB.shape)
    print("Label shape:", label.shape)

    break