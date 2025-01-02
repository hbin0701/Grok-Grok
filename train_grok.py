import os
import random
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from pathlib import Path

from model import Transformer
from utils import analyze_neuron_frequencies

@dataclass
class TrainingConfig:
    d_vocab: int
    seed: int
    lr: float
    weight_decay: float
    frac_train: float
    num_epochs: int
    stopping_thresh: float
    batch_size: int
    num_layers: int
    d_model: int
    batch_style: str
    n_ctx: int
    d_mlp: int
    num_heads: int
    d_head: int
    act_type: str
    use_ln: bool

class ArithmeticTrainer:
    def __init__(self, config: TrainingConfig, modulo: int, function_name: str):
        self.config = config
        self.modulo = modulo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize functions dictionary
        self.fns_dict = {
            'Task1': lambda x, y: (x + y) % modulo,                        # Original - Addition
            'Task2': lambda x, y: (x - y) % modulo,                        # New - Subtraction
            'Task3': lambda x, y: (x + y)**2 % modulo,                     # Original - Add then Square
            'Task4': lambda x, y: (x**2 + y**2) % modulo,                  # Original - Square then Add
            'Task5': lambda x, y: (x * pow(y, modulo-2, modulo)) % modulo, # New - Division
            'Task6': lambda x, y: (2*x*y) % modulo,           # New - Quadratic with cross term
            'Task7': lambda x, y: (x**3 + y**3) % modulo,          # New - Cubic with quadratic term
            'Task8': lambda x, y: (x + y)** 3 % modulo,
            'Task9': lambda x, y: (x*y) % modulo
        }
        self.fn = self.fns_dict[function_name]
        
        # Initialize model
        self.model = self._initialize_model()
        self.optimizer = self._initialize_optimizer()
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, 
            lambda step: min(step/10, 1)
        )

    def _initialize_model(self) -> Transformer:
        model = Transformer(
            num_layers=self.config.num_layers,
            d_vocab=self.config.d_vocab,
            d_model=self.config.d_model,
            d_mlp=self.config.d_mlp,
            d_head=self.config.d_head,
            num_heads=self.config.num_heads,
            n_ctx=self.config.n_ctx,
            act_type=self.config.act_type,
            use_cache=False,
            use_ln=self.config.use_ln
        )
        return model.to(self.device)

    def _initialize_optimizer(self) -> optim.AdamW:
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.98)
        )

    @staticmethod
    def cross_entropy_high_precision(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
        prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
        return -torch.mean(prediction_logprobs)

    def compute_loss_and_accuracy(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = data[:, :-1]
        labels = data[:, -1]
        
        try:
            logits = self.model(inputs)[:, -1]
        except:
            logits = self.model(inputs)['logits'][:, -1]
        
        loss = self.cross_entropy_high_precision(logits, labels)
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == labels).float().mean()
        
        return loss, accuracy

    def generate_dataset(self) -> Tuple[DataLoader, DataLoader]:
        def gen_pairs(frac_train: float, num: int, seed: int = 0) -> Tuple[List, List]:
            pairs = [(i, j, self.fn(i, j)) for i in range(num) for j in range(num)]
            random.seed(seed)
            random.shuffle(pairs)
            div = int(frac_train * len(pairs))
            return pairs[:div], pairs[div:]

        train_data, test_data = gen_pairs(self.config.frac_train, self.modulo, self.config.seed)
        
        train_tensor = torch.tensor(train_data, dtype=torch.long).to(self.device)
        test_tensor = torch.tensor(test_data, dtype=torch.long).to(self.device)
        
        train_dataset = TensorDataset(train_tensor)
        test_dataset = TensorDataset(test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
        
        return train_loader, test_loader

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        
        for batch in train_loader:
            batch = batch[0].to(self.device)
            loss, acc = self.compute_loss_and_accuracy(batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_acc += acc.item()
        
        return total_loss / len(train_loader), total_acc / len(train_loader)

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch[0].to(self.device)
                loss, acc = self.compute_loss_and_accuracy(batch)
                total_loss += loss.item()
                total_acc += acc.item()
        
        return total_loss / len(test_loader), total_acc / len(test_loader)

    def save_checkpoint(self, save_dir: str, run_name: str, 
                       train_losses: List[float], test_losses: List[float], 
                       epochs: List[int], state_dicts: List[Dict]) -> None:
        final_dict = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'epochs': epochs,
            'state_dicts': state_dicts,
            'model': self.model.state_dict(),
            'config': self.config.__dict__
        }
        
        save_path = os.path.join(save_dir, run_name, "full_run_data.pth")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(final_dict, save_path)

    def train(self, run_name: str, project_name: str) -> None:
                
        # Initialize wandb
        wandb.init(
            id=wandb.util.generate_id(),
            name=run_name,
            project=project_name
        )

        train_loader, test_loader = self.generate_dataset()
        train_losses, test_losses = [], []
        epochs_list, state_dicts = [], []

        for epoch in tqdm(range(self.config.num_epochs)):
            train_loss, train_acc = self.train_epoch(train_loader)
            test_loss, test_acc = self.evaluate(test_loader)
            
            self.scheduler.step()

            # Log metrics
            wandb.log({
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'lr': self.scheduler.get_last_lr()[0]
            }, step=epoch)

            # Save checkpoints
            if epoch % 10 == 0 and epoch != 0:
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                epochs_list.append(epoch)
                state_dicts.append(self.model.state_dict())
                
                res = analyze_neuron_frequencies(self.model, device=self.device)
                wandb.log(res, step=epoch)

            # Early stopping
            if test_loss < self.config.stopping_thresh:
                print(f"Early stopping at epoch {epoch}")
                print(f"Final train accuracy: {train_acc:.4f}")
                print(f"Final test accuracy: {test_acc:.4f}")
                break

        ckpt_dir = Path(__file__).parent / "ckpts"
        ckpt_dir.mkdir(exist_ok=True, parents=True)

        # Save final results
        self.save_checkpoint(
            ckpt_dir,
            run_name,
            train_losses,
            test_losses,
            epochs_list,
            state_dicts
        )

def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--project_name", type=str, default="Arith-Transfer-Again")
    parser.add_argument("--fn_name", type=str, default="Task1", 
                       choices=["Task1", "Task2", "Task3", "Task4", "Task5", "Task6", "Task7", "Task8", "Task9"])
    args = parser.parse_args()

    # Configuration
    config = TrainingConfig(
        d_vocab=114,  # p + 1
        seed=42,
        lr=1e-3,
        weight_decay=1,
        frac_train=0.3,
        num_epochs=50000,
        stopping_thresh=1e-9,
        batch_size=16384,
        num_layers=1,
        d_model=128,
        batch_style='full',
        n_ctx=3,
        d_mlp=512,  # 4*d_model
        num_heads=4,
        d_head=32,  # d_model//num_heads
        act_type='ReLU',
        use_ln=False
    )

    # Initialize trainer
    trainer = ArithmeticTrainer(config, modulo=113, function_name=args.fn_name)
    
    # Load checkpoint if provided
    if args.ckpt:
        ckpt = torch.load(args.ckpt)["model"]
        trainer.model.load_state_dict(ckpt)
        run_name = f"{args.ckpt.split('/')[-2]}->{args.fn_name}"
    else:
        run_name = args.fn_name
            
    trainer.train(run_name, args.project_name)

if __name__ == "__main__":
    main()