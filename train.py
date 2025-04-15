
import os
import yaml
import torch
import random
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from pre_processing import PreProcessing
from speaker_identification.config import Config
from speaker_identification.speaker_identification import SpeakerIdentificationModel
from utils.dataset import SpeakerDataset
from utils.sampler import FixedLengthBatchSampler

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(config_path):
    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    return Config(**raw_cfg)

def evaluate(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            wavs, lengths, lbls = [x.to(device) for x in batch]
            outputs = model(wavs, lengths)
            predicted = torch.argmax(outputs, dim=1)
            preds.extend(predicted.cpu().numpy())
            labels.extend(lbls.cpu().numpy())
    return accuracy_score(labels, preds)

def main(config_path):
    cfg = load_config(config_path)
    set_seed(cfg.training["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    preprocessor = PreProcessing(cfg)
    model = SpeakerIdentificationModel(cfg).to(device)

    dataset = SpeakerDataset(cfg, preprocessor)
    train_loader = DataLoader(
        dataset.train_dataset,
        batch_sampler=FixedLengthBatchSampler(dataset.train_dataset, cfg.training["batch_size"]),
        collate_fn=dataset.collate_fn,
        num_workers=2
    )

    test_loader = DataLoader(
        dataset.test_dataset,
        batch_sampler=FixedLengthBatchSampler(dataset.test_dataset, cfg.training["batch_size"]),
        collate_fn=dataset.collate_fn,
        num_workers=2
    )

    optimizer = optim.AdamW(model.parameters(), lr=cfg.training["learning_rate"], weight_decay=cfg.training["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training["epochs"])
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    os.makedirs(cfg.training["save_path"], exist_ok=True)

    for epoch in range(cfg.training["epochs"]):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training['epochs']}"):
            wavs, lengths, lbls = [x.to(device) for x in batch]
            optimizer.zero_grad()
            outputs = model(wavs, lengths, lbls)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Training Loss: {avg_loss:.4f}")

        if (epoch + 1) % cfg.training["eval_interval"] == 0:
            acc = evaluate(model, test_loader, device)
            print(f"Validation Accuracy: {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(cfg.training["save_path"], "best_model.pth"))
                print("✅ Best model saved.")

        scheduler.step()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    main(args.config)
