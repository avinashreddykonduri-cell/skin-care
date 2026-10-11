import argparse
import csv
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from models.network import OBSERVATIONS, SKIN_TYPES, SkinObservationNet

class LabeledImages(Dataset):
    def __init__(self, csv_path: Path):
        self.root = csv_path.parent
        with csv_path.open(newline="", encoding="utf-8") as source: self.rows = list(csv.DictReader(source))
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(self.root / row["image"]).convert("RGB")
        labels = torch.tensor([float(row[key]) for key in OBSERVATIONS])
        return self.transform(image), labels, SKIN_TYPES.index(row["skin_type"])

def train(data_csv: Path, output: Path, epochs: int, batch_size: int):
    loader = DataLoader(LabeledImages(data_csv), batch_size=batch_size, shuffle=True, num_workers=0)
    model = SkinObservationNet(pretrained=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    binary_loss, type_loss = torch.nn.BCEWithLogitsLoss(), torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for images, observations, skin_types in loader:
            prediction, type_prediction = model(images)
            loss = binary_loss(prediction, observations) + type_loss(type_prediction, skin_types)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); total += loss.item()
        print(f"epoch {epoch + 1}/{epochs}: loss={total / max(1, len(loader)):.4f}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "observations": OBSERVATIONS, "skin_types": SKIN_TYPES}, output)
    print(f"Saved model checkpoint to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the DermaCare ResNet-18 transfer-learning baseline.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/skin_model.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(); train(args.csv, args.output, args.epochs, args.batch_size)