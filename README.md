# Arith_transfer
Investigating Task Transfer on Modular Arithmetics

## Training
Run training using `train_grok.py` with the following arguments:
- `--fn_name`: Function to train (ADD, ADD_SQUARE, or SQUARE_ADD)
- `--project_name`: Name for wandb tracking
- `--checkpoint`: Path to resume from checkpoint (optional)

Example:
```bash
python train_grok.py --fn_name ADD --project_name my_experiment
```

## Visualization
Use `visualize.ipynb` to analyze results and create plots.