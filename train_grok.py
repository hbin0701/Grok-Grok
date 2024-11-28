import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import wandb
import os
import random
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM
from model import Transformer
from argparse import ArgumentParser

args = ArgumentParser()
args.add_argument("--ckpt", type=str, default="")
args.add_argument("--project_name", type=str, default="Arith-Transfer")
args.add_argument("--fn_name", type=str, default="ADD", choices=["ADD", "ADD_SQUARE", "SQUARE_ADD"])
args = args.parse_args()

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Function & WanDB Setting
project_name = args.project_name
fn_name = args.fn_name
run_name =f"[Task: {fn_name}] from scratch"

p = 113
fns_dict = {'ADD': lambda x,y:(x+y) % p, 'ADD_SQUARE': lambda x,y:(x+y)**2 % p, 'SQUARE_ADD': lambda x,y:(x**2+y**2) % p}
fn = fns_dict[fn_name]

# Helper functions
def cuda_memory():
    print(torch.cuda.memory_allocated()/1e9)

def cross_entropy_high_precision(logits, labels):
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss

def full_loss_and_accuracy(model, data):
    inputs = data[:, :-1]
    labels = data[:, -1]
    try:
        logits = model(inputs)[:, -1]
    except:
        logits = model(inputs)['logits'][:, -1]
    
    loss = cross_entropy_high_precision(logits, labels)
    
    # Calculate accuracy
    predictions = torch.argmax(logits, dim=-1)
    accuracy = (predictions == labels).float().mean()
    
    return loss, accuracy

def gen_train_test(frac_train, num, seed=0):
    pairs = [(i, j, fn(i, j)) for i in range(num) for j in range(num)]
    random.seed(seed)
    random.shuffle(pairs)
    div = int(frac_train*len(pairs))
    return pairs[:div], pairs[div:]

# WANDB Setting
logging_dir = os.path.join(os.getcwd(), "log")
os.makedirs(logging_dir, exist_ok=True)
logging_dir = "/home/hyeonbin/reason/Self-Explore/wandb"

wandb_id = wandb.util.generate_id()

wandb.init(id=wandb_id, 
           dir=logging_dir,
           name=run_name,
           project=project_name)

# HyperParameters
d_vocab = p + 1
seed = 42
lr = 1e-3
weight_decay = 1
frac_train = 0.3
num_epochs = 30000
stopping_thresh = 1e-9
batch_size = 8192

num_layers = 1
d_model = 128 #@param
batch_style = 'full'
n_ctx = 3
d_mlp = 4*d_model
num_heads = 4
assert d_model % num_heads == 0
d_head = d_model//num_heads
act_type = 'ReLU' #@param ['ReLU', 'GeLU']
use_ln = False

# Make a config
config = {
    "d_vocab": d_vocab,
    "seed": seed,
    "lr": lr,
    "weight_decay": weight_decay,
    "frac_train": frac_train,
    "num_epochs": num_epochs,
    "stopping_thresh": stopping_thresh,
    "batch_size": batch_size,
    "num_layers": num_layers,
    "d_model": d_model,
    "batch_style": batch_style,
    "n_ctx": n_ctx,
    "d_mlp": d_mlp,
    "num_heads": num_heads,
    "d_head": d_head,
    "act_type": act_type,
    "use_ln": use_ln
}

torch.manual_seed(seed)

# Generate train and test data
train_data, test_data = gen_train_test(frac_train, p, seed)

# Convert to tensors
train_tensor = torch.tensor(train_data, dtype=torch.long).to('cuda')
test_tensor = torch.tensor(test_data, dtype=torch.long).to('cuda')


# Create DataLoaders
train_dataset = TensorDataset(train_tensor)
test_dataset = TensorDataset(test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Load model
model = Transformer(num_layers=num_layers, d_vocab=d_vocab, d_model=d_model, d_mlp=d_mlp, d_head=d_head, num_heads=num_heads, n_ctx=n_ctx, act_type=act_type, use_cache=False, use_ln=use_ln)
model.to("cuda")

if args.ckpt != "":
    ckpt = torch.load(args.ckpt)["model"]
    model.load_state_dict(ckpt)

# Optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step/10, 1))

train_losses = []
test_losses = []
epochs = []
state_dicts = []

# Training loop
for epoch in tqdm(range(num_epochs)):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for batch in train_loader:
        batch = batch[0].to('cuda')
        loss, acc = full_loss_and_accuracy(model, batch)
        train_loss += loss.item()
        train_acc += acc.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch[0].to('cuda')
            loss, acc = full_loss_and_accuracy(model, batch)
            test_loss += loss.item()
            test_acc += acc.item()
    
    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    
    scheduler.step()

    # wandb log
    wandb.log({
        'train_loss': train_loss,
        'test_loss': test_loss,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'lr': scheduler.get_last_lr()[0]
    }, step=epoch)

    # Saving state dict every 100 epochs.
    if epoch % 100 == 0:
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        epochs.append(epoch)
        state_dicts.append(model.state_dict()) # Is it okay like this? wouldn't it consume GPU?
    
    # Stop Criteria.
    if test_loss < stopping_thresh:
        print(f"Early stopping at epoch {epoch}")
        print(f"Final train accuracy: {train_acc:.4f}")
        print(f"Final test accuracy: {test_acc:.4f}")
        break
    

# Gather all and save.
final_dict = {
    'train_losses': train_losses,
    'test_losses': test_losses,
    'epochs': epochs,
    'state_dicts': state_dicts,
    'model': model.state_dict(),
    'config': config
}

save_dir = os.getcwd() + "/" + run_name
os.makedirs(save_dir, exist_ok=True)
torch.save(final_dict, save_dir + "/full_run_data.pth")

print("Training completed")
print(f"Final train accuracy: {train_acc:.4f}")
print(f"Final test accuracy: {test_acc:.4f}")