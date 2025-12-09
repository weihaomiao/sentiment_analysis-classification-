# Sentiment Analysis with Linear Classifiers

A machine learning project implementing and comparing three linear classification algorithms for sentiment analysis on product reviews. The project demonstrates end-to-end ML pipeline development including feature engineering, model implementation, hyperparameter tuning, and evaluation.

## Overview

This project implements three linear classifiers from scratch:
- **Perceptron Algorithm**: Basic classification learning algorithm
- **Average Perceptron**: Improved variant that averages weights over iterations
- **Pegasos**: Primal Estimated sub-GrAdient SOlver for SVM (stochastic gradient descent with regularization)

The models are trained on product review data to classify sentiment as positive (+1) or negative (-1).

## Features

- **Bag-of-Words Feature Extraction**: Converts text reviews into numerical feature vectors with optional stopword removal
- **Multiple Classifier Implementations**: Three linear classification algorithms implemented from scratch
- **Hyperparameter Tuning**: Simple tunning for optimal model parameters
- **Model Evaluation**: Comprehensive accuracy metrics on training, validation, and test sets
- **Feature Analysis**: Identification of most explanatory word features for model interpretability

## Project Structure

```
sentiment_analysis/
├── data/                    # Dataset files
│   ├── reviews_train.tsv   # Training data
│   ├── reviews_val.tsv     # Validation data
│   ├── reviews_test.tsv    # Test data
│   └── stopwords.txt       # Stopword list
├── src/                     # Source code
│   ├── main.py             # Main execution script
│   ├── project1.py         # Core ML algorithms implementation
│   └── utils.py            # Utility functions (data loading, plotting, tuning)
└── test/                    # Unit tests
    └── test.py              # Test suite for all algorithms
```

## Requirements

- Python 3.x
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/weihaomiao/sentiment_analysis-classification-.git
cd sentiment_analysis-classification-
```

2. Install dependencies:
```bash
pip install numpy matplotlib
```

## Usage

### Running the Main Pipeline

Execute the main script to train models, perform hyperparameter tuning, and evaluate on test data:

```bash
python src/main.py
```

This will:
1. Load and preprocess the review data
2. Extract bag-of-words features
3. Train all three classifiers
4. Perform hyperparameter tuning
5. Evaluate on test set and display results
6. Identify most explanatory features

### Running Tests

Run the test suite to verify algorithm implementations:

```bash
python test/test.py
```

## Technical Details

### Feature Engineering

- **Text Preprocessing**: Lowercasing, punctuation/digit handling
- **Stopword Removal**: Optional filtering of common words
- **Feature Extraction**: Binary bag-of-words representation (word presence/absence)

### Algorithms

1. **Perceptron**: Simple learning algorithm that updates weights on misclassified examples
2. **Average Perceptron**: Maintains running average of weights to improve generalization
3. **Pegasos**: Stochastic sub-gradient descent with L2 regularization for SVM optimization

### Hyperparameter Tuning

- **T (Iterations)**: Number of passes through training data
- **L (Lambda)**: Regularization parameter for Pegasos algorithm
- Grid search performed over multiple parameter combinations

## Results

The pipeline automatically selects the best-performing model based on validation accuracy and reports:
- Training, validation, and test accuracies for each algorithm
- Optimal hyperparameters for each model
- Top 10 most explanatory word features

## Code Quality

- Clean, modular code structure with separation of concerns
- Comprehensive docstrings and type hints
- Unit tests for all core algorithms
- Path-agnostic file handling for portability
