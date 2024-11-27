# Skin Cancer Classification

Training deep learning models on the HAM10000 dataset for skin cancer classification across 7 different diagnostic categories. The dataset contains 10,015 dermatoscopic images.

## Setup and Usage
(Note: Below setup uses conda)
1. Run `make setup_environment` to create the conda env with the correct python version and pip-tools
2. Activate the newly created conda env with `conda activate syde577`
3. Run `make install_deps` to install the project dependencies

**Important**: 

4. Run `make setup_data` to download and prepare the HAM10000 dataset. 

### To Add A New Dependency
Add it to the python/requirements/requirements.in file and run make generate_new_requirements and commit the new requirements.txt file it generates.


## Project Structure
```
.
├── configs/               # Training configurations
│   └── config.yaml       # Base configuration template
├── src/
│   ├── data/            # Dataset implementations
│   ├── models/          # Model architectures
│   └── training/        # Training logic
├── experiments/         # Training scripts
└── scripts/            # Utility scripts
```

## Running Experiments
1. Create a config and training script that you will use for the experiment. You can just copy the existing templates found in `config/config.yaml` and `experiments/experiment_template.py` for example:

```bash
# Copy the template config
cp configs/config.yaml configs/my_experiment.yaml

# Copy the template training script
cp experiments/experiment_template.py experiments/my_experiment.py
```

2. Modify the config

```yaml
# configs/my_experiment.yaml
training:
  epochs: 100
  criterion:
    name: "CrossEntropyLoss"  # Can use any PyTorch loss
    params: {}
  optimizer:
    name: "Adam"              # Can use any PyTorch optimizer
    params:
      lr: 0.001
      weight_decay: 0.0001

data:
  batch_size: 32
  num_workers: 4
  image_size: 224
```


3. Run your training script (from root directory)

```bash
python -m experiments.my_experiment --config configs/my_experiment.yaml --output_dir outputs/my_experiment
```

## Adding New Models
1. Create a new model in `src/models/`
2. Import it in your experiment script as mentioned above. 

Example:

```python
from src.models.your_model import YourModel
from src.training.trainer import Trainer

def main():
    # ... setup code ...
    
    # Initialize your model
    model = YourModel(num_classes=7)
    
    # Train using the existing trainer
    trainer = Trainer(config)
    trainer.train(model, train_loader, val_loader)
```

