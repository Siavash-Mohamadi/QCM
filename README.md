# QCM Project

This project involves the implementation of various models to analyze QCM (Quartz Crystal Microbalance) data. The QCM class provides multiple functions to preprocess the data, fit different models, and visualize the results.

## Project Structure

- `QCM.ipynb`: Jupyter notebook demonstrating the usage of the QCM class.
- `data/`: Directory containing the input data file (`c8 high3.xlsx`).
- `src/`: Directory containing the Python class implementation (`qcm_class.py`).
- `requirements.txt`: List of dependencies required for the project.
- `README.md`: Project description and instructions.
- `LICENSE`: License information.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Make sure you have Python installed on your system. It's recommended to use a virtual environment.

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. Navigate to the project directory:

    ```bash
    cd QCM_Project
    ```

3. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    ```

4. Activate the virtual environment:

    - On Windows:

        ```bash
        venv\Scripts\activate
        ```

    - On macOS/Linux:

        ```bash
        source venv/bin/activate
        ```

5. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. **Run the Jupyter Notebook**:

    Open Jupyter Notebook and run `QCM.ipynb` to see the example usage of the QCM class.

    ```bash
    jupyter notebook QCM.ipynb
    ```

2. **QCM Class Implementation**:

    The QCM class, located in `src/qcm_class.py`, contains various methods for data preprocessing, model fitting, and plotting. 

### Files Description

- **`QCM.ipynb`**: A Jupyter notebook that demonstrates how to use the QCM class to process data, fit models, and visualize results.
- **`data/c8 high3.xlsx`**: The input data file used in the Jupyter notebook.
- **`src/qcm_class.py`**: The Python file containing the QCM class implementation.
- **`requirements.txt`**: A file listing all the dependencies required for the project.
- **`README.md`**: This file, providing an overview and instructions for the project.
- **`LICENSE`**: License information for the project.

### QCM Class Overview

The `QCM` class provides the following functionality:

- **Initialization**: Load and preprocess data.
- **Model Fitting**: Fit various models to the data.
  - Avrami
  - Pseudo-First-Order
  - Lagged_Pseudo-First-Order
  - Elovich
  - Exponential Growth
  - Boltzmann Sigmoidal
  - Double Exponential
  - Pseudo-Second-Order
- **Plotting**: Visualize raw data and fitted models.
- **Baseline Correction**: Perform baseline correction on the data.

### Example Usage

Here’s a brief example of how to use the QCM class:

```python
import pandas as pd
from src.qcm_class import QCM

# Load data
df = pd.read_excel('path to qcm.xlsx')

# Initialize QCM object
qcm = QCM(data=df, smoothing=True, time_span=1100, lowpass_freq=0.001)

# Plot raw data
qcm.plot_raw()

# Baseline correction
qcm.baseline_correction(870, 880)

# Pre-cut data
qcm.pre_cut(860, 915)

# Further baseline processing
qcm.baseline()
qcm.baseline_cut()

# Fit and plot model
qcm.fit_and_plot_2(save=False, model_name='Boltzmann Sigmoidal (Free Start)')
