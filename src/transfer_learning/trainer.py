import torch, os
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrainingArguments
from rich.console import Console
from rich.table import Table
from IPython.display import clear_output
import time
import math

class Trainer:
    """
    Custom Trainer for training and evaluating Transformer models.
    """
    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset,
        training_args: TrainingArguments,
        compute_metrics=None,
        collate_fn=None
    ):
        """
        Initialize the Trainer with the required components.
        Args:
            model: The Transformer model to train.
            train_dataset: The dataset for training.
            eval_dataset: The dataset for evaluation.
            tokenizer: Tokenizer used for pre/post-processing.
            training_args: Hugging Face TrainingArguments for configuration.
            compute_metrics: Function to compute evaluation metrics.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_args = training_args
        self.compute_metrics = compute_metrics
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.collate_fn = collate_fn
        self.metrics_history = []
        self.console = Console()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
            eps=training_args.adam_eps,
        )
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            factor=training_args.factor,
            patience=training_args.patience,
            verbose=True,
        )
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=training_args.src_pad_idx)
    
        # Ensure output directory exists
        os.makedirs(training_args.output_dir, exist_ok=True)

    @staticmethod
    def epoch_time(start_time: int, end_time: int):
        """
        Calculate elapsed time in minutes and seconds.
        """
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time % 60)
        return elapsed_mins, elapsed_secs
        
    def train(self):
        """
        Train the model using the training dataset.
        """
        self.model.train()
        self.model.to(self.device)
        start_time = time.time()

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.training_args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            drop_last=True
        )

        global_step = 0
        for epoch in range(self.training_args.num_train_epochs):
            epoch_loss = 0
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{self.training_args.num_train_epochs}"):
                global_step += 1
                src_ids = batch["input_ids"].to(self.device)
                tgt_ids = batch["labels"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(src_ids, tgt_ids[:, :-1])
                
                # Reshape for loss computation
                logits = outputs.view(-1, outputs.shape[-1])
                targets = tgt_ids[:, 1:].contiguous().view(-1)

                loss = self.criterion(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_args.max_grad_norm)
                self.optimizer.step()

                epoch_loss += loss.item()

                # Evaluate after eval_steps if specified
                if self.training_args.eval_steps and global_step % self.training_args.eval_steps == 0:
                    self.evaluate(train_loss=epoch_loss / global_step, epoch=int(epoch+1))
                    self.save_model(step=global_step)

            # Scheduler step
            self.scheduler.step(epoch_loss)

            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            _time = f"{epoch_mins}m {epoch_secs}s"
            print(f"Epoch {epoch} | Time: {epoch_mins}m {epoch_secs}s")

            # Evaluate after each epoch
            if not self.training_args.eval_steps:
                train_loss = epoch_loss / len(train_loader)
                self.save_model()
                self.evaluate(train_loss=train_loss, epoch=int(epoch+1))

    def evaluate(self, train_loss=None, epoch=None):
        """
        Evaluate the model on the validation dataset.
        """
        self.console.print("[bold green]Evaluation started...[/bold green]", end="\r")
        self.model.eval()
        self.model.to(self.device)
        
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.training_args.per_device_eval_batch_size,
            collate_fn=self.collate_fn,
            drop_last=True
        )

        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"Evaluation", ncols=10):
                src_ids = batch["input_ids"].to(self.device)
                tgt_ids = batch["labels"].to(self.device)

                outputs = self.model(src_ids, tgt_ids[:, :-1])
                logits = outputs.view(-1, outputs.shape[-1])
                targets = tgt_ids[:, 1:].contiguous().view(-1)

                loss = self.criterion(logits, targets)
                total_loss += loss.item()

                # Store predictions and labels
                predictions = torch.argmax(logits, dim=-1).cpu().tolist()
                labels = targets.cpu().tolist()
                all_predictions.extend(predictions)
                all_labels.extend(labels)

        #print(f"Validation Loss: {total_loss / len(eval_loader)}")
        metrics = {}
        eval_loss = total_loss / len(eval_loader)
        metrics["epoch"]  = epoch
        metrics["train_loss"] = train_loss
        metrics["eval_loss"]  = eval_loss
        metrics["PPL"]        = math.exp(train_loss)
        
        
        # Compute additional metrics if provided
        if self.compute_metrics:
            metrics_ = self.compute_metrics(all_predictions, all_labels)
            metrics.update(metrics_)
        
        # Append metrics to history and display
        self.display_metrics(metrics)

    def display_metrics(self, metrics):
        # Add the current metrics to the history
        self.metrics_history.append(metrics)

        # Clear the notebook cell output
        self.console.clear(home=True)
        clear_output(wait=True)

        # Create a new table
        table = Table(title="Metrics Summary", show_header=True, header_style="bold magenta")
        
        # Add the column headers (Epoch + all metric names)
        for key in metrics.keys():
            table.add_column(key, justify="center")

        # Add rows for all recorded metrics
        for _, epoch_metrics in enumerate(self.metrics_history, start=1):
            row = []
            for key in metrics.keys():
                value = epoch_metrics.get(key, "N/A")
                row.append(f"{value:.4f}" if isinstance(value, (float, int)) else str(value))
            table.add_row(*row)

        # Print the updated table
        self.console.print(table)

    def save_model(self, step=None):
        """
        Save the model and its weights to the output directory.
        """
        save_path = os.path.join(self.training_args.output_dir, f"checkpoint-{step}" if step else "final")
        os.makedirs(save_path, exist_ok=True)

        # Save model weights
        torch.save(self.model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))

        # Save the model configuration if applicable
        if hasattr(self.model, "config"):
            with open(os.path.join(save_path, "config.json"), "w") as f:
                f.write(self.model.config.to_json_string())