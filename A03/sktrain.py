"""
Training script using sklearn implementations for comparison and validation
This serves as a baseline to compare against from-scratch implementations
"""

import numpy as np
import argparse
import logging
import json
import os
import config
from preprocessing import DataPreprocessor
from visualization import Visualizer

# sklearn implementations
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import shap

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, 'sklearn_training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Silence verbose SHAP logging
logging.getLogger("shap").setLevel(logging.WARNING)


def evaluate_sklearn_model(model, X_test: np.ndarray, y_test: np.ndarray,
                           model_name: str = 'Model') -> dict:
    """
    Evaluate sklearn model on test data
    
    Args:
        model: Trained sklearn model
        X_test: Test features
        y_test: Test labels
        model_name: Name of model for logging
        
    Returns:
        Dictionary of metrics
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluating {model_name}")
    logger.info(f"{'='*50}")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # For binary classification
    if len(np.unique(y_test)) == 2:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro'
        )
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist()
    }
    
    # Log results
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    # Detailed classification report
    logger.info(f"\nClassification Report:")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    
    return metrics


def test_robustness_sklearn(model, X_test: np.ndarray, y_test: np.ndarray,
                            model_name: str = 'Model') -> dict:
    """
    Test model robustness to adversarial noise
    
    Args:
        model: Trained sklearn model
        X_test: Test features
        y_test: Test labels
        model_name: Name of model for logging
        
    Returns:
        Dictionary of robustness metrics
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Testing Robustness: {model_name}")
    logger.info(f"{'='*50}")
    
    robustness_results = {
        'clean_accuracy': float(accuracy_score(y_test, model.predict(X_test))),
        'noise_levels': config.NOISE_LEVELS,
        'gaussian_accuracies': [],
        'adversarial_accuracies': []
    }
    
    # Test with Gaussian noise
    logger.info("\nTesting with Gaussian noise:")
    for noise_level in config.NOISE_LEVELS:
        noise = np.random.normal(0, noise_level, X_test.shape)
        X_noisy = X_test + noise
        y_pred = model.predict(X_noisy)
        accuracy = accuracy_score(y_test, y_pred)
        robustness_results['gaussian_accuracies'].append(float(accuracy))
        logger.info(f"  Noise level {noise_level}: Accuracy = {accuracy:.4f}")
    
    # Test with adversarial noise (random direction)
    logger.info("\nTesting with adversarial noise:")
    for noise_level in config.NOISE_LEVELS:
        # Simple adversarial perturbation (random direction)
        noise = np.random.randn(*X_test.shape)
        noise = noise / (np.linalg.norm(noise, axis=1, keepdims=True) + 1e-10)
        X_adv = X_test + noise_level * noise
        y_pred = model.predict(X_adv)
        accuracy = accuracy_score(y_test, y_pred)
        robustness_results['adversarial_accuracies'].append(float(accuracy))
        logger.info(f"  Epsilon {noise_level}: Accuracy = {accuracy:.4f}")
    
    return robustness_results


def train_sklearn_logistic(X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           tune_hyperparams: bool = False) -> LogisticRegression:
    """
    Train sklearn logistic regression model
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        tune_hyperparams: Whether to perform hyperparameter tuning
        
    Returns:
        Trained model
    """
    logger.info("\n" + "="*50)
    logger.info("Training Sklearn Logistic Regression")
    logger.info("="*50)
    
    if tune_hyperparams:
        # Hyperparameter tuning with GridSearchCV
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['lbfgs', 'saga'],
            'max_iter': [1000, 2000]
        }
        
        model = LogisticRegression(random_state=config.RANDOM_STATE)
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='accuracy',
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        model = grid_search.best_estimator_
    else:
        # Train with default/config parameters
        model = LogisticRegression(
            C=1.0 / config.LOGISTIC_REGRESSION['lambda_'],
            max_iter=config.LOGISTIC_REGRESSION['max_iterations'],
            solver='lbfgs',
            random_state=config.RANDOM_STATE,
            verbose=1 if config.VERBOSE else 0
        )
        model.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_accuracy = accuracy_score(y_val, model.predict(X_val))
    logger.info(f"\nValidation Accuracy: {val_accuracy:.4f}")
    
    return model


def train_sklearn_svm(X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     kernel: str = 'rbf',
                     tune_hyperparams: bool = False) -> SVC:
    """
    Train sklearn SVM model
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        kernel: Kernel type
        tune_hyperparams: Whether to perform hyperparameter tuning
        
    Returns:
        Trained model
    """
    logger.info("\n" + "="*50)
    logger.info(f"Training Sklearn SVM with {kernel} kernel")
    logger.info("="*50)
    
    if tune_hyperparams:
        # Hyperparameter tuning
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }
        if kernel == 'poly':
            param_grid['degree'] = [2, 3, 4]
        
        model = SVC(kernel=kernel, random_state=config.RANDOM_STATE, probability=True)
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='accuracy',
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        model = grid_search.best_estimator_
    else:
        # Train with default/config parameters
        model = SVC(
            C=config.SVM['C'],
            kernel=kernel,
            gamma=config.SVM['gamma'],
            degree=config.SVM['degree'] if kernel == 'poly' else 3,
            random_state=config.RANDOM_STATE,
            probability=True,
            verbose=config.VERBOSE
        )
        model.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_accuracy = accuracy_score(y_val, model.predict(X_val))
    logger.info(f"\nValidation Accuracy: {val_accuracy:.4f}")
    
    return model


def main():
    """Main training function using sklearn"""
    parser = argparse.ArgumentParser(description='Train models using sklearn')
    parser.add_argument('--model', type=str, default='both',
                       choices=['logistic', 'svm', 'both'],
                       help='Model to train')
    parser.add_argument('--binary', action='store_true',
                       help='Binary classification (benign vs malicious)')
    parser.add_argument('--kernel', type=str, default='rbf',
                       choices=['linear', 'poly', 'rbf'],
                       help='Kernel for SVM')
    parser.add_argument('--tune', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip preprocessing (use preprocessed data)')
    parser.add_argument('--preprocessed-data', type=str,
                       default='processed_data.npz',
                       help='Path to preprocessed data file')
    parser.add_argument('--skip-visualization', action='store_true',
                       help='Skip visualization')
    parser.add_argument('--skip-interpretability', action='store_true',
                       help='Skip interpretability analysis')
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("SKLEARN BASELINE TRAINING PIPELINE")
    logger.info("="*70)
    
    # Load or preprocess data
    if args.skip_preprocessing and os.path.exists(args.preprocessed_data):
        logger.info(f"Loading preprocessed data from {args.preprocessed_data}")
        data = np.load(args.preprocessed_data)
        X = data['X']
        y = data['y']
        feature_names = data['feature_names'].tolist() if 'feature_names' in data else None
        class_names = data['class_names'].tolist() if 'class_names' in data else None
    else:
        # Preprocess data
        preprocessor = DataPreprocessor()
        
        try:
            data = preprocessor.load_data()
            
            # EDA
            logger.info("\nPerforming Exploratory Data Analysis...")
            eda_stats = preprocessor.explore_data(data)
            
            # Preprocess
            X, y = preprocessor.preprocess(data, binary=args.binary)
            feature_names = preprocessor.feature_names
            class_names = preprocessor.class_names
            
            # Save preprocessed data
            save_path = os.path.join(config.DATA_DIR, 'processed_data.npz')
            np.savez(save_path, X=X, y=y,
                    feature_names=feature_names,
                    class_names=class_names)
            logger.info(f"Saved preprocessed data to {save_path}")
                
        except FileNotFoundError as e:
            logger.error(f"Error: {e}")
            logger.error("\nPlease download the CICIDS2017 dataset and place it in the data/ directory")
            return
    
    # Split data
    logger.info("\nSplitting data into train/val/test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=config.TEST_SIZE + config.VALIDATION_SIZE,
        random_state=config.RANDOM_STATE, stratify=y
    )
    
    val_ratio = config.VALIDATION_SIZE / (config.TEST_SIZE + config.VALIDATION_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_ratio,
        random_state=config.RANDOM_STATE, stratify=y_temp
    )
    
    logger.info(f"Train set: {X_train.shape}")
    logger.info(f"Validation set: {X_val.shape}")
    logger.info(f"Test set: {X_test.shape}")
    
    # Train models
    models = {}
    
    if args.model in ['logistic', 'both']:
        lr_model = train_sklearn_logistic(X_train, y_train, X_val, y_val, args.tune)
        models['Sklearn Logistic Regression'] = lr_model
    
    if args.model in ['svm', 'both']:
        svm_model = train_sklearn_svm(X_train, y_train, X_val, y_val, args.kernel, args.tune)
        models[f'Sklearn SVM ({args.kernel})'] = svm_model
    
    # Evaluate models
    logger.info("\n" + "="*70)
    logger.info("MODEL EVALUATION")
    logger.info("="*70)
    
    all_metrics = {}
    for model_name, model in models.items():
        metrics = evaluate_sklearn_model(model, X_test, y_test, model_name)
        all_metrics[model_name] = metrics
        
        # Test robustness
        robustness = test_robustness_sklearn(model, X_test, y_test, model_name)
        all_metrics[model_name]['robustness'] = robustness
    
    # Save metrics
    metrics_path = os.path.join(config.RESULTS_DIR, 'sklearn_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Visualization
    if not args.skip_visualization:
        logger.info("\n" + "="*70)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("="*70)
        
        visualizer = Visualizer()
        
        for model_name, model in models.items():
            safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
            
            # Confusion matrix
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            # Ensure class_names is set
            if class_names is None:
                class_names_to_use = [str(i) for i in range(cm.shape[0])]
            else:
                class_names_to_use = class_names
                
            visualizer.plot_confusion_matrix(
                cm, class_names=class_names_to_use,
                title=f'Confusion Matrix - {model_name}',
                filename=f'sklearn_confusion_matrix_{safe_name}.png'
            )
            
            # ROC curve (for binary classification)
            if len(np.unique(y_test)) == 2:
                y_proba = model.predict_proba(X_test)[:, 1]
                visualizer.plot_roc_curve(
                    y_test, y_proba,
                    title=f'ROC Curve - {model_name}',
                    filename=f'sklearn_roc_curve_{safe_name}.png'
                )
            
            # Robustness plot
            robustness = all_metrics[model_name]['robustness']
            visualizer.plot_robustness_analysis(
                robustness['noise_levels'],
                robustness['gaussian_accuracies'],
                title=f'Robustness to Gaussian Noise - {model_name}',
                filename=f'sklearn_robustness_gaussian_{safe_name}.png'
            )
    
    # Interpretability with SHAP
    if not args.skip_interpretability:
        logger.info("\n" + "="*70)
        logger.info("MODEL INTERPRETABILITY (SHAP)")
        logger.info("="*70)
        
        for model_name, model in models.items():
            logger.info(f"\nGenerating SHAP analysis for {model_name}...")
            
            try:
                # Use SHAP's TreeExplainer or KernelExplainer
                if hasattr(model, 'coef_'):
                    # Linear model - use coefficients directly
                    explainer = shap.LinearExplainer(model, X_train[:100])
                else:
                    # Use KernelExplainer for SVM
                    explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])
                
                shap_values = explainer.shap_values(X_test[:50])
                
                # Summary plot
                safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
                shap.summary_plot(
                    shap_values, X_test[:50],
                    feature_names=feature_names,
                    show=False
                )
                import matplotlib.pyplot as plt
                plt.tight_layout()
                plt.savefig(os.path.join(config.INTERPRETABILITY_DIR, 
                                        f'sklearn_shap_summary_{safe_name}.png'))
                plt.close()
                logger.info(f"SHAP summary plot saved for {model_name}")
                
            except Exception as e:
                logger.warning(f"SHAP analysis failed for {model_name}: {e}")
    
    logger.info("\n" + "="*70)
    logger.info("SKLEARN TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"\nResults saved to: {config.RESULTS_DIR}")
    logger.info(f"Logs saved to: {config.LOGS_DIR}")


if __name__ == '__main__':
    main()
