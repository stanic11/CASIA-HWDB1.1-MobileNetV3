import json
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from MobileNet import MobileNetV3
from sklearn.metrics import recall_score, average_precision_score
import torchvision.transforms.functional as F
from myAPP import App
import tkinter as tk
isTrain = False

class ResizePad:
    def __init__(self, size=96, fill=255):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = F.resize(img, (new_h, new_w), interpolation=transforms.InterpolationMode.BICUBIC)
        pad_w = self.size - new_w
        pad_h = self.size - new_h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        img = F.pad(img, padding, fill=self.fill, padding_mode='constant')
        return img

class Trainer:
    def __init__(self, params):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} device.")

        self.batch_size = params.get("batch_size")
        self.epochs = params.get("epochs")
        self.patience = params.get("patience", 16)
        self.num_classes = params.get("num_classes", 100)
        self.save_path = params.get("save_path", './MobileNetV3.pth')

        self.train_data_path = params["train_data_dir"]
        self.test_data_path = params["test_data_dir"]

        self.data_transform = {
            "train": transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomRotation(10),  # 添加旋转
                transforms.RandomAffine(0, scale=(0.8, 1.2), shear=15),  # 添加缩放/剪切变化
                ResizePad(size=96, fill=255),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ]),
            "test": transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                ResizePad(size=96, fill=255),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        }

        self._init_dataloader()
        self._init_model()

    def _init_dataloader(self):
        self.train_dataset = datasets.ImageFolder(root=self.train_data_path, transform=self.data_transform["train"])
        self.test_dataset = datasets.ImageFolder(root=self.test_data_path, transform=self.data_transform["test"])
        print(self.train_dataset.classes)

        nw = os.cpu_count()
        print(f"Using {nw} dataloader workers every process.")

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=nw, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=nw, pin_memory=True)

        print(f"Using {len(self.train_dataset)} images for training, {len(self.test_dataset)} images for validation.")

    def _init_model(self):
        self.model = MobileNetV3(num_classes=self.num_classes)
        model_weight_path = os.path.join("weights", "mobilenet_v3_large-8738ca79.pth")

        pre_weights = torch.load(model_weight_path, map_location="cpu", weights_only=True)
        pre_dict = {k: v for k, v in pre_weights.items()
                    if k in self.model.state_dict() and self.model.state_dict()[k].numel() == v.numel()}
        self.model.load_state_dict(pre_dict, strict=False)

        for param in self.model.features.parameters():
            param.requires_grad = False

        self.model.to(self.device)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=0.01, weight_decay=1e-4)
        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.9)

    def train(self):
        best_acc = 0.0
        no_improve_epochs = 0

        train_loss_list = []
        val_loss_list = []
        val_acc_list = []
        val_map_list = []
        val_recall_list = []

        train_steps = len(self.train_loader)

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for step, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if step % 50 == 49:
                    print(f"Epoch [{epoch+1}/{self.epochs}] Step [{step+1}/{train_steps}] Loss: {loss:.3f}")

            epoch_train_loss = running_loss / train_steps
            train_loss_list.append(epoch_train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            all_preds = []
            all_labels = []
            all_scores = []

            with torch.no_grad():
                for images, labels in self.test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    val_loss += self.loss_function(outputs, labels).item()
                    preds = torch.max(outputs, 1)[1]
                    correct += (preds == labels).sum().item()
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_scores.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())

            epoch_val_loss = val_loss / len(self.test_loader)
            epoch_val_acc = correct / len(self.test_dataset)
            val_loss_list.append(epoch_val_loss)
            val_acc_list.append(epoch_val_acc)

            epoch_val_recall = recall_score(all_labels, all_preds, average='macro')
            epoch_val_map = average_precision_score(all_labels, all_scores, average='macro')

            val_recall_list.append(epoch_val_recall)
            val_map_list.append(epoch_val_map)

            print(f"[Epoch {epoch+1}] Train Loss: {epoch_train_loss:.3f}  Val Loss: {epoch_val_loss:.3f}  Val Acc: {epoch_val_acc:.3f}  Val Recall: {epoch_val_recall:.3f}  Val MAP: {epoch_val_map:.3f}")
            self.scheduler.step()

            # 保存最佳模型
            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                no_improve_epochs = 0
                torch.save(self.model.state_dict(), self.save_path)
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= self.patience:
                    print("Early stop")
                    break

        print("Finished Training.")
        self._save_metrics(train_loss_list, val_loss_list, val_acc_list, val_recall_list, val_map_list)

    def _save_metrics(self, train_loss_list, val_loss_list, val_acc_list, val_recall_list, val_map_list):
        os.makedirs('data', exist_ok=True)
        metrics_df = pd.DataFrame({
            'epoch': list(range(1, len(train_loss_list) + 1)),
            'train_loss': train_loss_list,
            'val_loss': val_loss_list,
            'val_acc': val_acc_list,
            'val_recall': val_recall_list,
            'val_map': val_map_list
        })
        metrics_df.to_csv('data/metrics.csv', index=False)
        print("Saved training metrics to 'data/metrics.csv'.")

if __name__ == "__main__":
    with open("params.json", 'r') as f:
        params = json.load(f)
    trainer = Trainer(params)
    
    if(isTrain == False):
        root = tk.Tk()
        app = App(root)
        root.mainloop()
    else:
        trainer.train()