# QCM

This project involves the implementation of various models to analyze QCM (Quartz Crystal Microbalance) data. The QCM class provides multiple functions to preprocess the data, fit different models, and visualize the results.

## Project Structure

- `QCM.ipynb`: Jupyter notebook demonstrating the usage of the QCM class.
- `src/`: Directory containing the Python class implementation (`qcm_class.py`).
- `requirements.txt`: List of dependencies required for the project.
- `README.md`: Project description and instructions.
- `LICENSE`: License information.


### Files Description

- **`QCM.ipynb`**: A Jupyter notebook that demonstrates how to use the QCM class to process data, fit models, and visualize results.
- **`src/qcm_class.py`**: The Python file containing the QCM class implementation.
- **`requirements.txt`**: A file listing all the dependencies required for the project.
- **`README.md`**: This file, providing an overview and instructions for the project.
- **`LICENSE`**: License information for the project.

## QCM Class Overview

The `QCM` class provides the following functionality:

- **Initialization**: Load and preprocess data, with options for smoothing and setting a time span for analysis.
- **Model Fitting**: Fit various models to the data.
  - Avrami
  - Pseudo-First-Order
  - Lagged Pseudo-First-Order
  - Elovich
  - Exponential Growth
  - Boltzmann Sigmoidal (Free Start)
  - Boltzmann Sigmoidal (Fixed Start)
  - Boltzmann Sigmoidal (B=1, Free Start)
  - Boltzmann Sigmoidal (B=1, Fixed Start)
  - Double Exponential
  - Double Exponential (Non-symmetric)
  - Pseudo-Second-Order
- **Plotting**: Visualize raw data, fitted models, and change points.
- **Baseline Correction**: Perform baseline correction on the data.
- **Change Point Detection**: Detect change points in the data using the BottomUp method.
- **Data Filtering**: Apply a low-pass filter to smooth the data.
- **Cutting Data**: Cut the data based on detected change points to focus on specific segments.
- **Residual Analysis**: Plot residuals to evaluate the fit quality.

