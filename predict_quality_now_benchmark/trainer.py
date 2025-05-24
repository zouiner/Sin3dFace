import os
import torch
from tqdm import tqdm
import torch.optim as optim
from model import ConfidenceModel
from torch.utils.data import DataLoader, random_split
from dataloader import AutoStructuredDataset

from torch.utils.tensorboard import SummaryWriter


class TrainerBase:
    def __init__(self, configs):
        self.configs = configs
        self.device = configs['device']

        self.batch_size = configs['batch_size']
        self.num_epochs = configs['num_epochs']
        self.learning_rate = configs['learning_rate']
        self.results_dir = configs['results_dir']
        self.dataset_dir = configs['dataset_dir']
        self.mesh_feat_dim = configs['mesh_feat_dim']
        self.save_log = configs['save_log']
        self.thresholds = configs.get('thresholds', [1.15, 1.15, 1.15])
        self.val_size = configs['val_size']

        self.start_epoch = 0

        if self.configs['resume']:
            self.resume = self.configs['results_dir']
        else:
            self.resume = None

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        if self.results_dir:
            self.results_output = f"{self.results_dir}/confidence_regression_model.pth"
        else:
            self.results_output = 'confidence_regression_model.pth'

        # Save hyperparameters to a config.txt file
        config_lines = [
            f"dataset_dir: {self.dataset_dir}",
            f"batch_size: {self.batch_size}",
            f"num_epochs: {self.num_epochs}",
            f"learning_rate: {self.learning_rate}",
            f"mesh_feat_dim: {self.mesh_feat_dim}"
        ]

        with open(f"{self.results_dir}/config.txt", 'w') as f:
            f.write("\n".join(config_lines))

        # Intialize model
        self.model = self._build_model()
        self.model.to(self.device)
        self.optimizer = self._build_optimizer()
        if self.resume:
            self.load_model(self.resume)
        self.writer = SummaryWriter(log_dir=os.path.join(self.results_dir, 'tf_logs'))
        self.loss_fn = torch.nn.BCELoss()


    
    def _build_model(self):
        return ConfidenceModel(
            mesh_feat_dim=self.configs['mesh_feat_dim'],
            fusion_hidden_dim=self.configs['fusion_hidden_dim']
        )
    
    def _build_optimizer(self):

        return optim.Adam(self.model.parameters(), lr=self.configs['learning_rate'])

    def data_loader(self, dataset_dir):
        dataset = AutoStructuredDataset(dataset_dir)

        total_size = len(dataset)
        val_size = int(self.val_size * total_size)
        train_size = total_size - val_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def load_model(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} not found.")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.start_epoch = checkpoint.get('epoch', 0)

        print(f"✅ Loaded model from {checkpoint_path}, resuming at epoch {self.start_epoch}")

    def save_model(self, epoch=None):
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch if epoch is not None else self.epoch,
        }

        if epoch is not None:
            save_path = os.path.join(self.results_dir, 'model')
            os.makedirs(save_path, exist_ok=True)
            checkpoint_path = os.path.join(save_path, f"model_{epoch}.pth")
        else:
            checkpoint_path = self.results_output

        torch.save(state, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def train(self, dataloader):
        self.model.train()
        total_loss = 0.0
        total_correct = torch.zeros(3, device=self.device)
        total_all_correct = 0
        total_samples = 0

        for batch in tqdm(dataloader, desc="Training"):
            sr_img = batch["image"].to(self.device)
            mesh_feat = batch["mesh_feat"].to(self.device)
            labels = batch["labels"].to(self.device)  # [B, 3]

            self.optimizer.zero_grad()
            outputs = self.model(sr_img, mesh_feat)  # [B, 3]
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Accuracy calculation
            preds = (outputs >= 0.5).int()
            targets = labels.int()

            total_correct += (preds == targets).sum(dim=0)
            total_all_correct += ((preds == targets).all(dim=1)).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {self.epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")

        for i, name in enumerate(["median", "mean", "std"]):
            acc = 100.0 * total_correct[i].item() / total_samples
            print(f"[TRAIN] {name} Accuracy: {total_correct[i].item()}/{total_samples} = {acc:.2f}%")
            self.writer.add_scalar(f'Accuracy/train_{name}', acc, self.epoch)

        overall_acc = 100.0 * total_all_correct / total_samples
        print(f"[TRAIN] Overall Accuracy: {total_all_correct}/{total_samples} = {overall_acc:.2f}%")
        self.writer.add_scalar('Accuracy/train_overall', overall_acc, self.epoch)
        self.writer.add_scalar('Loss/train', avg_loss, self.epoch)

        return avg_loss


    def validate(self, dataloader, phase="val"):
        self.model.eval()
        total_loss = 0.0
        total_correct = torch.zeros(3, device=self.device)
        total_all_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"[{phase.upper()}] Validation"):
                sr_img = batch["image"].to(self.device)
                mesh_feat = batch["mesh_feat"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(sr_img, mesh_feat)  # [B, 3]
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()

                preds = (outputs >= 0.5).int()
                true = labels.int()

                total_correct += (preds == true).sum(dim=0)
                total_all_correct += ((preds == true).all(dim=1)).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        for i, name in enumerate(["median", "mean", "std"]):
            correct_i = total_correct[i].item()
            acc = 100.0 * correct_i / total_samples
            print(f"[{phase.upper()}] {name} Accuracy: {correct_i}/{total_samples} = {acc:.2f}%")
            self.writer.add_scalar(f'Accuracy/{phase}_{name}', acc, self.epoch)

        overall_acc = 100.0 * total_all_correct / total_samples
        print(f"[{phase.upper()}] Overall Accuracy: {total_all_correct}/{total_samples} = {overall_acc:.2f}%")
        self.writer.add_scalar(f'Accuracy/{phase}_overall', overall_acc, self.epoch)
        self.writer.add_scalar(f'Loss/{phase}', avg_loss, self.epoch)

        print(f"[{phase.upper()}] Loss: {avg_loss:.4f}, Overall Accuracy: {overall_acc:.2f}%")
        return avg_loss, overall_acc

    def run(self):
        print("Starting training...")

        self.data_loader(self.dataset_dir)
        for self.epoch in range(self.start_epoch, self.num_epochs):
            print(f"\nEpoch {self.epoch+1}/{self.num_epochs}:")
            avg_train_loss = self.train(self.train_loader)
            avg_val_loss, overall_acc = self.validate(self.val_loader)

            print(f"Epoch {self.epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Overall Accuracy = {overall_acc:.2f}%")


        self.save_model()

        print(f"Model saved to {self.results_output}")
        self.writer.close()

    
    def inference(self, input_dirs, save_predictions=True):
        """
        Run inference on new data from input_dirs.
        
        Args:
            input_dirs (list[str]): List of folder paths containing sr_img and 3d_obj
            save_predictions (bool): Whether to save predictions to a text file

        Returns:
            List of predictions
        """
        self.model.eval()
        dataset = AutoStructuredDataset(input_dirs)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        predictions = []
        image_names = []
        
        # Accuracy counters (optional)
        correct = torch.zeros(3)
        all_correct = 0
        count = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference"):
                sr_img = batch['image'].to(self.device)
                mesh_feat = batch['mesh_feat'].to(self.device)

                outputs = self.model(sr_img, mesh_feat)  # [batch_size, 3]
                probs = outputs.cpu().numpy()
                preds = (probs >= 0.5).astype(int)  # shape: [batch_size, 3]

                predictions.extend(preds.tolist())
                image_names.extend([str(p.name) for p in batch['image_path']]) if 'image_path' in batch else None
                

                true = batch["labels"].int().cpu()
                correct += (preds == true).sum(dim=0)
                all_correct += ((preds == true).all(dim=1)).sum().item()
                count += preds.size(0)

        print("\nInference Accuracy Summary:")
        for i, name in enumerate(["median", "mean", "std"]):
            correct_i = correct[i].item()
            acc = 100.0 * correct_i / count
            print(f" - {name} acc: {correct_i}/{count} = {acc:.2f}%")

        overall_acc = 100.0 * all_correct / count
        print(f" - all metrics correct: {all_correct}/{count} = {overall_acc:.2f}%")

        if save_predictions:
            output_file = os.path.join(self.results_dir, "inference_predictions.txt")
            with open(output_file, 'w') as f:
                for i, pred in enumerate(preds):
                    labels = ["good" if x == 0 else "bad" for x in pred]
                    f.write(f"{image_names[i]}: {' '.join(labels)}\n")

            print(f"Inference results saved to {output_file}")

        return predictions