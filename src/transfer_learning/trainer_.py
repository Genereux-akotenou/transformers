import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import TrainingArguments
from tqdm import tqdm

class Trainer:
    """
    Custom Trainer for Transformers.
    """
    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset,
        tokenizer,
        training_args: TrainingArguments,
        compute_metrics=None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.compute_metrics = compute_metrics
        self.optimizer = torch.optim.Adam(model.parameters(), 
                                        lr=training_args.learning_rate, 
                                        weight_decay=training_args.weight_decay,
                                        eps=training_args.adam_eps)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                 verbose=True,
                                                 factor=training_args.factor,
                                                 patience=training_args.patience)
        self.criterion = nn.CrossEntropyLoss(ignore_index=training_args.src_pad_idx)

    def train(model, iterator, optimizer, criterion, clip):
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            optimizer.zero_grad()
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            epoch_loss += loss.item()
            print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())
    
        return epoch_loss / len(iterator)

    def train(self):
        self.model.train()
        dataloader = DataLoader(self.train_dataset, batch_size=self.training_args.per_device_train_batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        for epoch in range(self.training_args.num_train_epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
                src_ids = batch["input_ids"].to(device)
                tgt_ids = batch["labels"].to(device)

                self.optimizer.zero_grad()
                outputs, _, _ = self.model(src_ids, tgt_ids)
                print(outputs.shape)
                #self.model.tgt_vocab_size
                logits = outputs.view(-1, self.model.tgt_vocab_size)
                loss = torch.nn.CrossEntropyLoss()(logits, tgt_ids.view(-1))
                loss.backward(retain_graph=True)
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1} | Loss: {total_loss / len(dataloader)}")
            self.evaluate()

    def evaluate(self):
        self.model.eval()
        dataloader = DataLoader(self.eval_dataset, batch_size=self.training_args.per_device_eval_batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        total_eval_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                src_ids = batch["input_ids"].to(device)
                tgt_ids = batch["labels"].to(device)

                outputs, _, _ = self.model(src_ids, tgt_ids)
                print(outputs.shape)
                logits = outputs.view(-1, self.model.tgt_embedding.num_embeddings)
                loss = torch.nn.CrossEntropyLoss()(logits, tgt_ids.view(-1))
                total_eval_loss += loss.item()

                # Store predictions and labels for metric computation
                predictions = torch.argmax(logits, dim=-1).cpu().tolist()
                labels = tgt_ids.view(-1).cpu().tolist()
                all_predictions.extend(predictions)
                all_labels.extend(labels)

        print(f"Validation Loss: {total_eval_loss / len(dataloader)}")
        if self.compute_metrics:
            metrics = self.compute_metrics(all_predictions, all_labels)
            print(metrics)