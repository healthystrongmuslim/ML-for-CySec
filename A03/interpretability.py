"""
Model interpretability using SHAP and LIME
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from typing import List
import config

logger = logging.getLogger(__name__)


class ModelInterpreter:
    """
    Model interpretability using SHAP and LIME
    """
    
    def __init__(self, model, feature_names: List[str] = None, 
                 class_names: List[str] = None, save_dir: str = None):
        """
        Initialize interpreter
        
        Args:
            model: Trained model with predict and predict_proba methods
            feature_names: Names of features
            class_names: Names of classes
            save_dir: Directory to save interpretability plots
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.save_dir = save_dir if save_dir else config.INTERPRETABILITY_DIR
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.shap_explainer = None
        self.lime_explainer = None
    
    def explain_with_shap(self, X_train: np.ndarray, X_test: np.ndarray,
                         n_samples: int = None,
                         plot_type: str = 'summary') -> None:
        """
        Explain model predictions using SHAP
        
        Args:
            X_train: Training data (background dataset)
            X_test: Test data to explain
            n_samples: Number of samples to use for explanation
            plot_type: Type of SHAP plot ('summary', 'bar', 'waterfall', 'force')
        """
        try:
            import shap
        except ImportError:
            logger.error("SHAP not installed. Install with: pip install shap")
            return
        
        logger.info("Generating SHAP explanations...")
        
        # Subsample if needed
        if n_samples is None:
            n_samples = min(config.SHAP_SAMPLES, X_test.shape[0])
        
        X_test_sample = X_test[:n_samples]
        X_train_sample = X_train[:min(100, X_train.shape[0])]
        
        # Create explainer
        # For custom models, use KernelExplainer
        def model_predict(X):
            return self.model.predict_proba(X)
        
        self.shap_explainer = shap.KernelExplainer(model_predict, X_train_sample)
        
        # Compute SHAP values
        logger.info("Computing SHAP values (this may take a while)...")
        shap_values = self.shap_explainer.shap_values(X_test_sample)
        
        # Plot
        if plot_type == 'summary':
            plt.figure()
            shap.summary_plot(shap_values, X_test_sample, 
                            feature_names=self.feature_names,
                            show=False)
            filepath = os.path.join(self.save_dir, 'shap_summary.png')
            plt.savefig(filepath, bbox_inches='tight')
            logger.info(f"Saved SHAP summary plot to {filepath}")
            plt.close()
            
        elif plot_type == 'bar':
            plt.figure()
            shap.summary_plot(shap_values, X_test_sample,
                            feature_names=self.feature_names,
                            plot_type='bar', show=False)
            filepath = os.path.join(self.save_dir, 'shap_bar.png')
            plt.savefig(filepath, bbox_inches='tight')
            logger.info(f"Saved SHAP bar plot to {filepath}")
            plt.close()
        
        elif plot_type == 'waterfall':
            # Waterfall plot for first prediction
            plt.figure()
            shap.waterfall_plot(shap.Explanation(
                values=shap_values[0][0],
                base_values=self.shap_explainer.expected_value[0],
                data=X_test_sample[0],
                feature_names=self.feature_names
            ), show=False)
            filepath = os.path.join(self.save_dir, 'shap_waterfall.png')
            plt.savefig(filepath, bbox_inches='tight')
            logger.info(f"Saved SHAP waterfall plot to {filepath}")
            plt.close()
        
        logger.info("SHAP analysis complete")
    
    def explain_with_lime(self, X_train: np.ndarray, X_test: np.ndarray,
                        instance_idx: int = 0,
                        num_features: int = None) -> None:
        """
        Explain model predictions using LIME
        
        Args:
            X_train: Training data
            X_test: Test data to explain
            instance_idx: Index of instance to explain
            num_features: Number of top features to show
        """
        try:
            from lime import lime_tabular
        except ImportError:
            logger.error("LIME not installed. Install with: pip install lime")
            return
        
        logger.info("Generating LIME explanations...")
        
        if num_features is None:
            num_features = min(config.TOP_FEATURES, X_train.shape[1])
        
        # Create explainer
        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification'
        )
        
        # Explain instance
        instance = X_test[instance_idx]
        explanation = self.lime_explainer.explain_instance(
            instance,
            self.model.predict_proba,
            num_features=num_features
        )
        
        # Save explanation
        fig = explanation.as_pyplot_figure()
        filepath = os.path.join(self.save_dir, f'lime_explanation_{instance_idx}.png')
        plt.savefig(filepath, bbox_inches='tight')
        logger.info(f"Saved LIME explanation to {filepath}")
        plt.close()
        
        # Save as HTML
        html_filepath = os.path.join(self.save_dir, f'lime_explanation_{instance_idx}.html')
        explanation.save_to_file(html_filepath)
        logger.info(f"Saved LIME HTML report to {html_filepath}")
        
        logger.info("LIME analysis complete")
    
    def plot_feature_importance_comparison(self, X: np.ndarray,
                                          n_samples: int = 100) -> None:
        """
        Compare feature importance using model weights vs SHAP
        
        Args:
            X: Data samples
            n_samples: Number of samples for SHAP analysis
        """
        logger.info("Comparing feature importance methods...")
        
        # Get model-based importance
        model_importance = self.model.get_feature_importance()
        
        # Get SHAP-based importance (if available)
        try:
            import shap
            
            if self.shap_explainer is None:
                logger.warning("SHAP explainer not initialized. Call explain_with_shap first.")
                return
            
            # Use subset of data
            X_sample = X[:min(n_samples, X.shape[0])]
            shap_values = self.shap_explainer.shap_values(X_sample)
            
            # Average absolute SHAP values
            if isinstance(shap_values, list):
                # Multi-class: average across classes
                shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                shap_importance = np.abs(shap_values).mean(axis=0)
            
            # Normalize for comparison
            model_importance_norm = model_importance / model_importance.sum()
            shap_importance_norm = shap_importance / shap_importance.sum()
            
            # Plot comparison
            top_n = min(config.TOP_FEATURES, len(model_importance))
            top_indices = np.argsort(model_importance)[-top_n:][::-1]
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Model importance
            axes[0].barh(range(top_n), model_importance_norm[top_indices])
            if self.feature_names:
                axes[0].set_yticks(range(top_n))
                axes[0].set_yticklabels([self.feature_names[i] for i in top_indices])
            axes[0].set_xlabel('Normalized Importance')
            axes[0].set_title('Model Weight-Based Importance')
            
            # SHAP importance
            axes[1].barh(range(top_n), shap_importance_norm[top_indices])
            if self.feature_names:
                axes[1].set_yticks(range(top_n))
                axes[1].set_yticklabels([self.feature_names[i] for i in top_indices])
            axes[1].set_xlabel('Normalized Importance')
            axes[1].set_title('SHAP-Based Importance')
            
            plt.tight_layout()
            filepath = os.path.join(self.save_dir, 'importance_comparison.png')
            plt.savefig(filepath, bbox_inches='tight')
            logger.info(f"Saved importance comparison to {filepath}")
            plt.close()
            
        except ImportError:
            logger.warning("SHAP not available for comparison")
    
    def generate_local_explanations(self, X_test: np.ndarray, 
                                   y_test: np.ndarray,
                                   n_samples: int = 5) -> None:
        """
        Generate local explanations for multiple samples
        
        Args:
            X_test: Test data
            y_test: True labels
            n_samples: Number of samples to explain
        """
        logger.info(f"Generating local explanations for {n_samples} samples...")
        
        predictions = self.model.predict(X_test[:n_samples])
        
        for i in range(n_samples):
            logger.info(f"\nSample {i}:")
            logger.info(f"  True label: {y_test[i]}")
            logger.info(f"  Predicted label: {predictions[i]}")
            logger.info(f"  Match: {'✓' if y_test[i] == predictions[i] else '✗'}")
            
            # Generate LIME explanation
            try:
                self.explain_with_lime(X_test, X_test, instance_idx=i)
            except Exception as e:
                logger.warning(f"Could not generate LIME explanation: {e}")
        
        logger.info("Local explanations complete")


def main():
    """Main function for standalone interpretability analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Model interpretability analysis')
    parser.add_argument('--method', type=str, required=True,
                       choices=['shap', 'lime', 'both'],
                       help='Interpretability method')
    parser.add_argument('--model', type=str, required=True,
                       choices=['logistic', 'svm'],
                       help='Model to explain')
    
    args = parser.parse_args()
    
    logger.info(f"Running {args.method} interpretability for {args.model} model")
    logger.info("Load your trained model and data to generate explanations")


if __name__ == '__main__':
    main()
