# Setup Guide

## Prerequisites

- Python 3.9+
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/4yrg/mazemind.git
cd mazemind
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Virtual Environment

**Git Bash / Linux / macOS:**
```bash
source .venv/Scripts/activate
```

**Windows CMD:**
```bash
.venv\Scripts\activate.bat
```

**Windows PowerShell:**
```bash
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` in your terminal prompt.

### 4. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

This installs:
- `numpy` - Matrix operations for Q-tables
- `matplotlib` - Maze rendering and static plots
- `plotly` - Interactive charts
- `streamlit` - Web UI
- `pandas` - Data export
- `scipy` - Statistical tests
- `pytest` - Testing
- `jupyter` - Notebook support

### 5. Register Jupyter Kernel

```bash
python -m ipykernel install --user --name mazemind --display-name "Python (mazemind)"
```

This makes the venv available as a kernel in Jupyter notebooks.

### 6. Verify Installation

```bash
python -m pytest tests/ -v
```

All 45 tests should pass.

## VS Code Configuration

The repository includes `.vscode/settings.json` that automatically:
- Sets the Python interpreter to `.venv/Scripts/python.exe`
- Adds `src/` to the analysis path for autocomplete
- Configures Jupyter kernel preferences

After opening the project in VS Code:
1. Press `Ctrl+Shift+P` → "Reload Window"
2. Press `Ctrl+Shift+P` → "Python: Select Interpreter" → choose `.venv/Scripts/python.exe`
3. When opening notebooks, select the **Python (mazemind)** kernel

## Troubleshooting

### NumPy version conflict
If you see `NumPy 1.x cannot be run in NumPy 2.x`, ensure you're using the venv's Python:
```bash
.venv/Scripts/python.exe -c "import numpy; print(numpy.__version__)"
```

### Kernel not showing in Jupyter
Re-register the kernel:
```bash
python -m jupyter kernelspec uninstall mazemind -y
python -m ipykernel install --user --name mazemind --display-name "Python (mazemind)"
```

### Import errors in notebooks
Ensure the kernel is set to **Python (mazemind)**, not the system Python.
