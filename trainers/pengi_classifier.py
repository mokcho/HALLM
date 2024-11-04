import os
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm, trange
from alm.Pengi.wrapper import PengiWrapper as Pengi
from classifiers.linear_classifier import LinearClassifier


class PengiClassifierTrainer :
    def __init__(self, cfg):
        self.cfg = cfg
        self.pengi = Pengi(config=cfg.pengi.config)
        self.ckpt_path = os.path.join(cfg.data_dir, "checkpoints")
        os.makedirs(self.ckpt_path, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get Pengi Embeddings from obtained ckpt
        data_dir = os.path.join(self.cfg.data_dir, "Clotho/entailment")
        audio_paths, text_prompts, train_labels = self.load_data(os.path.join(data_dir,'clotho_development_gpt4_flatten.csv'))
        train_embeddings, missing_train = self.process_embeddings(audio_paths, text_prompts)
        val_audio_paths, val_text_prompts, val_labels = self.load_data(os.path.join(data_dir, 'clotho_validation_gpt4_flatten.csv'))
        val_embeddings, missing_val = self.process_embeddings(val_audio_paths, val_text_prompts)
        eval_audio_paths, eval_text_prompts, eval_labels = self.load_data(os.path.join(data_dir, 'clotho_evaluation_gpt4_flatten.csv'))
        eval_embeddings, missing_eval  = self.process_embeddings(eval_audio_paths, eval_text_prompts)

        train_labels = [label for i, label in enumerate(train_labels) if i not in missing_train]
        val_labels = [label for i, label in enumerate(val_labels) if i not in missing_val]
        eval_labels = [label for i, label in enumerate(eval_labels) if i not in missing_eval]

        train_embeddings_tensor = torch.stack(train_embeddings).squeeze(1)
        val_embeddings_tensor = torch.stack(val_embeddings).squeeze(1)
        eval_embeddings_tensor = torch.stack(eval_embeddings).squeeze(1)

        train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
        val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)
        test_labels_tensor = torch.tensor(eval_labels, dtype=torch.long)

        # Create DataLoaders
        train_dataset = TensorDataset(train_embeddings_tensor, train_labels_tensor)
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=cfg.batch_size,
                                       shuffle=True,
                                       num_workers=cfg.num_workers)
        val_dataset = TensorDataset(val_embeddings_tensor, val_labels_tensor)
        self.val_loader = DataLoader(val_dataset,
                                     batch_size=cfg.batch_size,
                                     num_workers=cfg.num_workers)
        test_dataset = TensorDataset(eval_embeddings_tensor, test_labels_tensor)
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=cfg.batch_size,
                                      num_workers=cfg.num_workers)

        assert len(train_embeddings[0]) == len(train_embeddings[1]), "Input Sizes are not Consistent!!"

        # Import Classifier Model, currently only Linear
        self.model = LinearClassifier(input_dim=len(train_embeddings[0])).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model,
            lr=cfg.lr
        )

        self.lr_scheduler = None #not implemented

        self.cur_epoch = 0

        # Resume Training
        ckpt_path = os.path.join(self.ckpt_path, f"{cfg.resume_ckpt}")
        if self.cfg.resume and os.path.exist(ckpt_path):
            print(f"Resume from checkpoint {ckpt_path}")
            self.load_checkpoint(self.cfg.resume_ckpt)

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        audio_paths = df['Audio file'].tolist()
        text_prompts = df['Hypothesis'].tolist()
        labels = df['Label'].tolist()
        return audio_paths, text_prompts, labels

    def process_embeddings(self, audio_paths, text_prompts):
        embeddings = []
        missing_files =[]
        audio_path = None
        for i in tqdm(range(len(audio_paths)), desc="Processing Embeddings"):
            if audio_paths[i] != audio_path:
                if os.path.isfile(audio_paths[i]):
                    _, audio_embeddings = self.pengi.get_audio_embeddings(audio_paths=[audio_paths[i]])
                    audio_path = audio_paths[i]
                else:
                    print(f"File not found: {audio_paths[i]}")
                    missing_files.append(i)
                    continue
            _, text_embeddings = self.pengi.get_prompt_embeddings(prompts=[text_prompts[i]])

            concat_embeddings = torch.cat((audio_embeddings, text_embeddings), dim=1)
            embeddings.append(concat_embeddings)
        return embeddings, missing_files

    def train(self):
        tq = trange(
            self.cur_epoch,
            self.cfg.epochs,
            desc='Training'
        )
        self.model.train()
        for i in tq :
            for embeddings in self.train_loader :
                embeddings, labels = embeddings.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(embeddings)
                loss = self.criterion(outputs, labels)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.lr_scheduler is not None :
                    self.lr_scheduler.step()

            val_f1, val_accuracy, val_precision, val_recall = self.evaluate(self.val_loader)
            print(f"Epoch [{self.cur_epoch + 1}/{self.cfg.epochs}], F1 : {val_f1:.4f} | Accuracy: {val_accuracy:.4f} | Precision : {val_precision:.4f} | Recall : {val_recall:.4f}")

        self.save_checkpoint(name=f"Pengi_Linear_{self.cur_epoch}_{self.cfg.lr}.pt")
        test_f1, test_accuracy, test_precision, test_recall = self.evaluate(self.test_loader)
        print(f"Test Scores _ F1 : {test_f1} | accuracy : {test_accuracy} | precision : {test_precision} | recall : {test_recall}")

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        val_predictions = []
        val_labels_list = []
        with torch.no_grad():
            for embeddings, labels in dataloader:
                embeddings = embeddings.to(self.device)
                outputs = self.model(embeddings)
                _, predicted = torch.max(outputs, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_labels_list.extend(labels.numpy())

        accuracy = accuracy_score(val_labels_list, val_predictions)
        precision = precision_score(val_labels_list, val_predictions, average='macro')
        recall = recall_score(val_labels_list, val_predictions, average='macro')
        f1 = f1_score(val_labels_list, val_predictions, average='macro')
        return f1, accuracy, precision, recall

    def save_checkpoint(self, name='latest.pt'):
        ckpt_path = os.path.join(self.ckpt_path, f"{name}")
        ckpt = {
            'cfg' : self.cfg,
            'model' : self.get_state_dict(self.model),
            'optimizer' : self.get_state_dict(self.optimizer),
            'lr_scheduler' : self.get_state_dict(self.lr_scheduler),
            'cur_epoch' : self.cur_epoch
        }
        torch.save(ckpt, ckpt_path)
        print(f"{ckpt_path} has been saved.")

    def load_checkpoint(self, name='latest.pt'):
        ckpt_path = os.path.join(self.ckpt_path, f"{name}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print(self.model.load_state_dict(ckpt['model']))
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        self.cur_epoch = ckpt['cur_epoch']

if __name__ == "__main__" :

    trainer = PengiClassifierTrainer(cfg="base")
    trainer.train()