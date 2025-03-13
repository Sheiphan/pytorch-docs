# Deep Learning with PyTorch Step-by-Step

A comprehensive guide to learning PyTorch for deep learning, from basic concepts to advanced applications.

## Overview

This repository contains the code and notebooks for the "Deep Learning with PyTorch Step-by-Step" tutorial series. It provides a structured approach to learning PyTorch, starting from the fundamentals and progressing to advanced deep learning techniques.

## Contents

- **Jupyter Notebooks**: Chapter-by-chapter tutorials (Chapter00.ipynb through Chapter11.ipynb)
- **Python Scripts**: Corresponding Python scripts for each chapter
- **StepByStep Library**: Custom implementation of neural network components that evolves throughout the tutorial
- **Datasets**: 
  - Rock-Paper-Scissors image dataset (rps.zip, rps-test-set.zip)
  - Pre-processed data files (.pth)
- **Utility Scripts**: Helper functions and configuration files

## Prerequisites

- Python 3.10+
- PyTorch 2.2.1+
- Other dependencies as listed in `environment.yml`

## Installation

### Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/pytorch-docs.git
cd pytorch-docs

# Create and activate the conda environment
conda env create -f environment.yml
conda activate pytorchbook
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/pytorch-docs.git
cd pytorch-docs

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt  # Note: You may need to create this from environment.yml
```

## Chapters Overview

- **Chapter 0**: Introduction and setup
- **Chapter 1**: PyTorch basics and tensor operations
- **Chapter 2**: Linear regression and gradient descent
- **Chapter 3**: Neural networks fundamentals
- **Chapter 4**: Training neural networks
- **Chapter 5**: Convolutional Neural Networks (CNNs)
- **Chapter 6**: Transfer learning and fine-tuning
- **Chapter 7**: Recurrent Neural Networks (RNNs)
- **Chapter 8**: Advanced RNN architectures
- **Chapter 9**: Natural Language Processing
- **Chapter 10**: Generative models
- **Chapter 11**: Advanced topics and techniques

## Usage

1. Start Jupyter Lab or Notebook:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

2. Navigate to the chapter you want to study and open the corresponding notebook.

3. Follow along with the explanations and code examples.

## StepByStep Library

The repository includes a custom `stepbystep` library that evolves throughout the tutorial:

- **v0.py**: Basic implementation
- **v1.py**: Added functionality
- **v2.py**: Enhanced features
- **v3.py**: Advanced components
- **v4.py**: Final version with all features

## Additional Resources

- **helpers.py**: Utility functions used across notebooks
- **config.py**: Configuration settings for the tutorials
- **plots/**: Visualization functions for each chapter
- **pretrained/**: Pre-trained models

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the work by Daniel Voigt Godoy
- PyTorch team for the excellent deep learning framework
- Contributors to the open-source libraries used in this project

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
