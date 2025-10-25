"""
Configuration file for the Intrusion Detection System
Contains all hyperparameters and settings for the project
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
INTERPRETABILITY_DIR = os.path.join(RESULTS_DIR, 'interpretability')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')

# Create directories if they don't exist
for dir_path in [DATA_DIR, RESULTS_DIR, PLOTS_DIR, INTERPRETABILITY_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Dataset settings
DATASET_FILES = [ # followed by class distribution comments
    # 'Monday-WorkingHours.pcap_ISCX.csv', # BAD.  has no positive/malicious class
    'Tuesday-WorkingHours.pcap_ISCX.csv', # GOOD. BENIGN:432074(96%)  FTP-Patator:7938  SSH-Patator:5897
    'Wednesday-workingHours.pcap_ISCX.csv',  # GOOD but huge.  BENIGN:440031(63%)  DoSHulk:231073  DoSGoldenEye:10293  DoSslowloris:5796  DoSSlowhttptest:5499  Heartbleed:11
    'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', # MEH.  BENIGN:168186(98%)  WebAttack�BruteForce:1507  WebAttack�XSS:652  WebAttack�SqlInjection:21
    # 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv', # BAD.  BENIGN:288566(99%)  Infiltration:36
    'Friday-WorkingHours-Morning.pcap_ISCX.csv', # MEH.  BENIGN:189067(99%)  Bot:1966
    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv' # GOOD.  BENIGN:127537(44%)  PortScan:158930  
    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv' # GOOD. BENIGN:97718(43%)  DDoS:128027
]

# Preprocessing settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Feature engineering
NORMALIZATION_METHOD = 'standard'  # 'standard', 'minmax', 'robust'
HANDLE_OUTLIERS = True
OUTLIER_METHOD = 'iqr'  # 'iqr', 'zscore', 'isolation_forest'
BALANCE_METHOD = 'smote'  # 'smote', 'undersample', 'oversample', 'none'

# Logistic Regression hyperparameters
LOGISTIC_REGRESSION = {
    'learning_rate': 0.01,
    'max_iterations': 1000,
    'tolerance': 1e-6,
    'regularization': 'l2',
    'lambda_': 0.01,
    'batch_size': 32,  # For mini-batch SGD
    'gradient_check': True,
    'gradient_check_epsilon': 1e-7
}

# SVM hyperparameters
SVM = {
    'C': 1.0,
    'kernel': 'rbf',  # 'linear', 'polynomial', 'rbf'
    'gamma': 'scale',  # 'scale', 'auto', or float value
    'degree': 3,  # For polynomial kernel
    'max_iterations': 1000,
    'tolerance': 1e-3,
    'learning_rate': 0.001
}

# Training settings
OPTIMIZATION_METHOD = 'gradient_descent'  # 'gradient_descent', 'sgd', 'mini_batch'
VERBOSE = True
EARLY_STOPPING = True
PATIENCE = 10

# Cross-validation
CV_FOLDS = 5

# Hyperparameter tuning
HYPERPARAMETER_TUNING = {
    'logistic_regression': {
        'learning_rate': [0.001, 0.01, 0.1],
        'lambda_': [0.001, 0.01, 0.1, 1.0],
        'max_iterations': [500, 1000, 2000]
    },
    'svm': {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'polynomial', 'rbf'],
        'gamma': [0.001, 0.01, 0.1, 'scale'],
        'degree': [2, 3, 4]  # Only for polynomial kernel
    }
}

# Interpretability settings
SHAP_SAMPLES = 100  # Number of samples for SHAP analysis
LIME_SAMPLES = 100  # Number of samples for LIME analysis
TOP_FEATURES = 20  # Number of top features to visualize

# Adversarial testing
ADVERSARIAL_EPSILON = 0.1  # FGSM epsilon value
NOISE_LEVELS = [0.01, 0.05, 0.1, 0.2]  # Gaussian noise levels

# Visualization settings
FIGURE_SIZE = (10, 6)
DPI = 300
COLOR_PALETTE = 'Set2'

# Logging
LOG_LEVEL = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Model saving
SAVE_MODELS = True
MODEL_DIR = os.path.join(RESULTS_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
