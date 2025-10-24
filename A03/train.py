"""
Main training script for the Intrusion Detection System
"""

import numpy as np
import argparse
import logging
import json
import os
from typing import Tuple
import config
import utils
from logistic_regression import LogisticRegression
from svm import SVM
from preprocessing import DataPreprocessor
from visualization import Visualizer
from interpretability import ModelInterpreter

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, 'training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                  model_name: str = 'Model') -> dict:
    """
    Evaluate model on test data
    
    Args:
        model: Trained model
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
    accuracy = utils.compute_accuracy(y_test, y_pred)
    
    # For binary classification
    if len(np.unique(y_test)) == 2:
        precision, recall, f1 = utils.compute_precision_recall_f1(y_test, y_pred, 'binary')
    else:
        precision, recall, f1 = utils.compute_precision_recall_f1(y_test, y_pred, 'macro')
    
    # Confusion matrix
    cm = utils.compute_confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist()
    }
    
    # Log results
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    return metrics


def test_robustness(model, X_test: np.ndarray, y_test: np.ndarray,
                   model_name: str = 'Model') -> dict:
    """
    Test model robustness to adversarial noise
    
    Args:
        model: Trained model
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
        'clean_accuracy': utils.compute_accuracy(y_test, model.predict(X_test)),
        'noise_levels': config.NOISE_LEVELS,
        'gaussian_accuracies': [],
        'adversarial_accuracies': []
    }
    
    # Test with Gaussian noise
    logger.info("\nTesting with Gaussian noise:")
    for noise_level in config.NOISE_LEVELS:
        X_noisy = utils.add_gaussian_noise(X_test, noise_level)
        y_pred = model.predict(X_noisy)
        accuracy = utils.compute_accuracy(y_test, y_pred)
        robustness_results['gaussian_accuracies'].append(accuracy)
        logger.info(f"  Noise level {noise_level}: Accuracy = {accuracy:.4f}")
    
    # Test with adversarial noise (FGSM-style with random gradients)
    logger.info("\nTesting with adversarial noise:")
    for noise_level in config.NOISE_LEVELS:
        # Simple adversarial perturbation (random direction)
        X_adv = utils.add_adversarial_noise(X_test, noise_level)
        y_pred = model.predict(X_adv)
        accuracy = utils.compute_accuracy(y_test, y_pred)
        robustness_results['adversarial_accuracies'].append(accuracy)
        logger.info(f"  Epsilon {noise_level}: Accuracy = {accuracy:.4f}")
    
    return robustness_results


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray) -> LogisticRegression:
    """
    Train logistic regression model
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        Trained model
    """
    logger.info("\n" + "="*50)
    logger.info("Training Logistic Regression")
    logger.info("="*50)
    
    # Initialize model
    model = LogisticRegression(
        learning_rate=config.LOGISTIC_REGRESSION['learning_rate'],
        max_iterations=config.LOGISTIC_REGRESSION['max_iterations'],
        tolerance=config.LOGISTIC_REGRESSION['tolerance'],
        lambda_=config.LOGISTIC_REGRESSION['lambda_'],
        batch_size=config.LOGISTIC_REGRESSION['batch_size'],
        optimization=config.OPTIMIZATION_METHOD,
        gradient_check=config.LOGISTIC_REGRESSION['gradient_check'],
        verbose=config.VERBOSE
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_accuracy = utils.compute_accuracy(y_val, model.predict(X_val))
    logger.info(f"\nValidation Accuracy: {val_accuracy:.4f}")
    
    return model


def train_svm(X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             kernel: str = 'rbf') -> SVM:
    """
    Train SVM model
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        kernel: Kernel type
        
    Returns:
        Trained model
    """
    logger.info("\n" + "="*50)
    logger.info(f"Training SVM with {kernel} kernel")
    logger.info("="*50)
    
    # Initialize model
    model = SVM(
        C=config.SVM['C'],
        kernel=kernel,
        gamma=config.SVM['gamma'],
        degree=config.SVM['degree'],
        max_iterations=config.SVM['max_iterations'],
        tolerance=config.SVM['tolerance'],
        learning_rate=config.SVM['learning_rate'],
        batch_size=config.SVM.get('batch_size', config.LOGISTIC_REGRESSION.get('batch_size', 32)),
        optimization=config.OPTIMIZATION_METHOD,
        verbose=config.VERBOSE
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_accuracy = utils.compute_accuracy(y_val, model.predict(X_val))
    logger.info(f"\nValidation Accuracy: {val_accuracy:.4f}")
    
    return model


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train intrusion detection models')
    parser.add_argument('--model', type=str, default='both',
                       choices=['logistic', 'svm', 'both'],
                       help='Model to train')
    parser.add_argument('--binary', action='store_true',
                       help='Binary classification (benign vs malicious)')
    parser.add_argument('--kernel', type=str, default='rbf',
                       choices=['linear', 'polynomial', 'rbf'],
                       help='Kernel for SVM')
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
    logger.info("INTRUSION DETECTION SYSTEM - TRAINING PIPELINE")
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
            if config.SAVE_MODELS:
                save_path = os.path.join(config.DATA_DIR, 'processed_data.npz')
                np.savez(save_path, X=X, y=y,
                        feature_names=feature_names,
                        class_names=class_names)
                logger.info(f"Saved preprocessed data to {save_path}")
                
        except FileNotFoundError as e:
            logger.error(f"Error: {e}")
            logger.error("\nPlease download the CICIDS2017 dataset and place it in the data/ directory")
            logger.error("Dataset available at: https://www.unb.ca/cic/datasets/ids-2017.html")
            return
    
    # Split data
    logger.info("\nSplitting data into train/val/test sets...")
    X_train, X_temp, y_train, y_temp = utils.train_test_split(
        X, y, test_size=config.TEST_SIZE + config.VALIDATION_SIZE,
        random_state=config.RANDOM_STATE
    )
    
    val_ratio = config.VALIDATION_SIZE / (config.TEST_SIZE + config.VALIDATION_SIZE)
    X_val, X_test, y_val, y_test = utils.train_test_split(
        X_temp, y_temp, test_size=1 - val_ratio,
        random_state=config.RANDOM_STATE
    )
    
    logger.info(f"Train set: {X_train.shape}")
    logger.info(f"Validation set: {X_val.shape}")
    logger.info(f"Test set: {X_test.shape}")
    
    # Train models
    models = {}
    
    if args.model in ['logistic', 'both']:
        lr_model = train_logistic_regression(X_train, y_train, X_val, y_val)
        models['Logistic Regression'] = lr_model
        
        if config.SAVE_MODELS:
            model_path = os.path.join(config.MODEL_DIR, 'logistic_regression.pkl')
            utils.save_model(lr_model, model_path)
    
    if args.model in ['svm', 'both']:
        svm_model = train_svm(X_train, y_train, X_val, y_val, kernel=args.kernel)
        models[f'SVM ({args.kernel})'] = svm_model
        
        if config.SAVE_MODELS:
            model_path = os.path.join(config.MODEL_DIR, f'svm_{args.kernel}.pkl')
            utils.save_model(svm_model, model_path)
    
    # Evaluate models
    logger.info("\n" + "="*70)
    logger.info("MODEL EVALUATION")
    logger.info("="*70)
    
    all_metrics = {}
    for model_name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name)
        all_metrics[model_name] = metrics
        
        # Test robustness
        robustness = test_robustness(model, X_test, y_test, model_name)
        all_metrics[model_name]['robustness'] = robustness
    
    # Save metrics
    metrics_path = os.path.join(config.RESULTS_DIR, 'metrics.json')
    utils.save_metrics(all_metrics, metrics_path)
    
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
            cm = utils.compute_confusion_matrix(y_test, y_pred)
            # Ensure class_names is set (fallback to numeric labels)
            if class_names is None:
                class_names_to_use = [str(i) for i in range(cm.shape[0])]
            else:
                class_names_to_use = class_names

            visualizer.plot_confusion_matrix(
                cm, class_names=class_names_to_use,
                title=f'Confusion Matrix - {model_name}',
                filename=f'confusion_matrix_{safe_name}.png'
            )
            
            # Training history
            if hasattr(model, 'loss_history'):
                visualizer.plot_training_history(
                    model.loss_history,
                    title=f'Training Loss - {model_name}',
                    filename=f'training_loss_{safe_name}.png'
                )
            
            # Feature importance
            importance = model.get_feature_importance()
            visualizer.plot_feature_importance(
                importance, feature_names=feature_names,
                title=f'Feature Importance - {model_name}',
                filename=f'feature_importance_{safe_name}.png'
            )
            
            # ROC curve (for binary classification)
            if len(np.unique(y_test)) == 2:
                y_proba = model.predict_proba(X_test)[:, 1]
                visualizer.plot_roc_curve(
                    y_test, y_proba,
                    title=f'ROC Curve - {model_name}',
                    filename=f'roc_curve_{safe_name}.png'
                )
            
            # Robustness plot
            robustness = all_metrics[model_name]['robustness']
            visualizer.plot_robustness_analysis(
                robustness['noise_levels'],
                robustness['gaussian_accuracies'],
                title=f'Robustness to Gaussian Noise - {model_name}',
                filename=f'robustness_gaussian_{safe_name}.png'
            )
    
    # Interpretability
    if not args.skip_interpretability:
        logger.info("\n" + "="*70)
        logger.info("MODEL INTERPRETABILITY")
        logger.info("="*70)
        
        for model_name, model in models.items():
            logger.info(f"\nGenerating interpretability analysis for {model_name}...")
            
            interpreter = ModelInterpreter(
                model,
                feature_names=feature_names,
                class_names=class_names
            )
            
            try:
                # SHAP analysis
                interpreter.explain_with_shap(
                    X_train[:100], X_test[:50],
                    plot_type='summary'
                )
            except Exception as e:
                logger.warning(f"SHAP analysis failed: {e}")
            
            try:
                # LIME analysis for a few samples
                interpreter.generate_local_explanations(X_test, y_test, n_samples=3)
            except Exception as e:
                logger.warning(f"LIME analysis failed: {e}")
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"\nResults saved to: {config.RESULTS_DIR}")
    logger.info(f"Models saved to: {config.MODEL_DIR}")
    logger.info(f"Logs saved to: {config.LOGS_DIR}")


if __name__ == '__main__':
    main()
