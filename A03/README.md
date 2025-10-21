# Intrusion Detection System using ML for CySec

## Overview
This project implements a robust binary and multi-class intrusion detection system using machine learning algorithms implemented from scratch. The system identifies malicious and benign traffic flows using the CICIDS2017 dataset.

## Features
- **Custom Implementations**: Logistic Regression and Support Vector Machine (SVM) built from scratch
- **Comprehensive EDA**: Detailed exploratory data analysis and preprocessing
- **Multiple Kernels**: SVM implementation with linear, polynomial, and RBF kernels
- **Interpretability**: SHAP/LIME analysis for model explainability
- **Robustness Testing**: Evaluation on clean and adversarially perturbed data
- **Hyperparameter Tuning**: Systematic optimization with documented methodology
- **Rich Visualizations**: Decision boundaries, confusion matrices, ROC curves, and more

## Project Structure
```
A03/
├── README.md                 # Project documentation
├── assignmentSummary.md      # Assignment requirements
├── requirements.txt          # Python dependencies
├── config.py                 # Configuration parameters
├── logistic_regression.py    # Logistic Regression implementation
├── svm.py                    # SVM implementation
├── preprocessing.py          # Data preprocessing and EDA
├── visualization.py          # Visualization utilities
├── interpretability.py       # SHAP/LIME analysis
├── train.py                  # Main training script
└── utils.py                  # Helper functions
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation
1. Clone the repository:
```bash
git clone https://github.com/healthystrongmuslim/ML-for-CySec
cd ML-for-CySec/A03
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Setup
1. Download the CICIDS2017 dataset from [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
2. Extract the dataset to a `data/` directory:
```bash
mkdir -p data
# Extract your downloaded CICIDS2017 files here
```

3. The expected directory structure:
```
data/
├── Monday-WorkingHours.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
├── Wednesday-workingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
└── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

## Usage

### Training Models
Run the main training script:
```bash
python train.py --model logistic --binary
python train.py --model svm --kernel rbf
python train.py --model both --multiclass
```

### Preprocessing and EDA
```bash
python preprocessing.py --explore
python preprocessing.py --preprocess --output processed_data.csv
```

### Visualization
```bash
python visualization.py --confusion-matrix --model logistic
python visualization.py --roc-curve --model svm
python visualization.py --decision-boundary
```

### Interpretability Analysis
```bash
python interpretability.py --method shap --model logistic
python interpretability.py --method lime --model svm
```

## Implementation Details

### Logistic Regression
- Implemented with gradient descent optimization
- Binary and multi-class classification support
- L2 regularization
- Gradient checking for validation
- Configurable learning rate and iterations

### Support Vector Machine (SVM)
- Hinge loss optimization
- Multiple kernel functions (linear, polynomial, RBF)
- Regularization parameter tuning
- From-scratch implementation without sklearn

### Preprocessing Pipeline
- Normalization (StandardScaler, MinMaxScaler)
- Outlier detection and handling
- Feature encoding for categorical variables
- Data balancing (SMOTE, undersampling, oversampling)
- Missing value imputation

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices
- ROC curves and AUC
- Training/validation loss curves
- Cross-validation scores

## Hyperparameter Tuning
The following hyperparameters are tuned using grid search:
- **Logistic Regression**: learning rate, regularization strength, iterations
- **SVM**: C parameter, kernel type, kernel parameters (gamma, degree)

Results are documented in `results/tuning_results.txt`

## Robustness Testing
Models are evaluated against:
- Clean test data
- Gaussian noise injection
- Adversarial perturbations (FGSM)

## Results
Training logs, performance metrics, and visualizations are saved in the `results/` directory:
- `results/metrics.json` - Performance metrics
- `results/plots/` - Visualization outputs
- `results/interpretability/` - SHAP/LIME explanations
- `results/logs/` - Training logs

## Bonus Features
- Stochastic Gradient Descent (SGD) implementation
- Mini-batch training support
- FGSM adversarial attack generator
- Model robustness analysis

## Authors
- Project developed as part of ML for CySec coursework

## License
This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## References
- CICIDS2017 Dataset: https://www.unb.ca/cic/datasets/ids-2017.html
- SHAP: https://github.com/slundberg/shap
- LIME: https://github.com/marcotcr/lime
