# Gyro Kan

A PyTorch Lightning project for training neural networks on the MNIST dataset with hyperparameter optimization using Optuna.

## Features

- PyTorch Lightning for scalable training
- Optuna for hyperparameter tuning
- TensorBoard integration for logging
- Optuna Dashboard for visualization
- Docker containerization with GPU support

## Prerequisites

- Python 3.8+
- uv (install via `curl -LsSf https://astral.sh/uv/install.sh | sh` on Unix or download from https://github.com/astral-sh/uv)
- NVIDIA GPU (optional, for GPU acceleration)

## Setup

### Using uv (Recommended for Local Development)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd gyro_kan
   ```

2. Create and activate a virtual environment:
   ```bash
   uv venv
   # On Windows: .venv\Scripts\activate
   # On Unix: source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

This sets up a local environment with all required packages (PyTorch, PyTorch Lightning, Optuna, TensorBoard, Jupyter).

### Building the Docker Image (Alternative)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd gyro_kan
   ```

2. Build the Docker image:
   ```bash
   docker build -t gyrokan .
   ```

### VS Code Integration with Dev Container (Alternative)

1. Open the project in VS Code.

2. When prompted, or via Command Palette (`Ctrl+Shift+P`), select "Dev Containers: Reopen in Container".

3. VS Code will build and open the project in the dev container with all dependencies installed.

The dev container includes:
- PyTorch with CUDA support
- All required Python packages (PyTorch Lightning, Optuna, etc.)
- GPU access (if available)
- Port forwarding for TensorBoard (6006) and Optuna Dashboard (8080)

## Running the Example Training Script

The `example_train.py` script demonstrates training a simple neural network on MNIST with Optuna hyperparameter optimization.

To run the training:

```bash
python example_train.py
```

This will:
- Download the MNIST dataset (if not already present)
- Perform hyperparameter optimization with Optuna
- Train the best model
- Save logs to `lightning_logs/`
- Generate Optuna plots
- Start the Optuna Dashboard at http://localhost:8080

You can monitor training progress with TensorBoard, if you are not using optuna dashboard, by running:

```bash
tensorboard --logdir lightning_logs/
```

Access TensorBoard at http://localhost:6006

## Project Structure

- `example_train.py`: Main training script with Optuna optimization
- `lightning_logs/`: Training logs and checkpoints
- `MNIST/`: Dataset directory

## Customization

- Modify hyperparameters in `example_train.py`
- Adjust model architecture in the `LitMNIST` class
- Change Optuna study parameters (number of trials, etc.)