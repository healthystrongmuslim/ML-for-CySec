"""
Data preprocessing and exploratory data analysis for CICIDS2017 dataset
"""

import numpy as np
import pandas as pd
from typing import Tuple
import logging
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import config

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocessor for CICIDS2017 dataset
    
    Handles:
    - Data loading and merging
    - Missing value handling
    - Feature encoding
    - Normalization
    - Outlier detection and removal
    - Class balancing
    """
    
    def __init__(self):
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.class_names = None
        
    def load_data(self, data_dir: str = None) -> pd.DataFrame:
        """
        Load CICIDS2017 dataset from CSV files
        
        Args:
            data_dir: Directory containing dataset files
            
        Returns:
            Merged DataFrame
        """
        if data_dir is None:
            data_dir = config.DATA_DIR
        
        logger.info(f"Loading data from {data_dir}")
        
        dfs = []
        for filename in config.DATASET_FILES:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                logger.info(f"Loading {filename}")
                df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
                dfs.append(df)
            else:
                logger.warning(f"File not found: {filepath}")
        
        if not dfs:
            raise FileNotFoundError(f"No dataset files found in {data_dir}")
        
        # Merge all dataframes
        data = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(data)} samples with {len(data.columns)} features")
        
        return data
    
    def explore_data(self, data: pd.DataFrame) -> dict:
        """
        Perform exploratory data analysis
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary with EDA statistics
        """
        logger.info("Performing exploratory data analysis...")
        
        eda_stats = {
            'n_samples': len(data),
            'n_features': len(data.columns),
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.to_dict(),
            'numeric_stats': {},
            'class_distribution': {}
        }
        
        # Find label column (usually named 'Label' or ' Label')
        label_col = None
        for col in data.columns:
            if 'label' in col.lower():
                label_col = col
                break
        
        if label_col:
            # Class distribution
            class_counts = data[label_col].value_counts()
            eda_stats['class_distribution'] = class_counts.to_dict()
            logger.info(f"\nClass distribution:\n{class_counts}")
            
            # Separate features and labels
            X = data.drop(columns=[label_col])
        else:
            logger.warning("Label column not found")
            X = data
        
        # Numeric statistics
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            eda_stats['numeric_stats'][col] = {
                'mean': X[col].mean(),
                'std': X[col].std(),
                'min': X[col].min(),
                'max': X[col].max(),
                'missing': X[col].isnull().sum()
            }
        
        logger.info(f"EDA complete. Found {len(numeric_cols)} numeric features")
        
        return eda_stats
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values...")
        
        # Check for missing values
        missing_counts = data.isnull().sum()
        if missing_counts.sum() == 0:
            logger.info("No missing values found")
            return data
        
        logger.info(f"Found {missing_counts.sum()} missing values")
        
        # For numeric columns, fill with median
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].isnull().any():
                median_value = data[col].median()
                data[col].fillna(median_value, inplace=True)
                logger.info(f"Filled {col} missing values with median: {median_value}")
        
        # For categorical columns, fill with mode
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if data[col].isnull().any():
                mode_value = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
                data[col].fillna(mode_value, inplace=True)
                logger.info(f"Filled {col} missing values with mode: {mode_value}")
        
        return data
    
    def handle_outliers(self, X: np.ndarray, method: str = 'iqr') -> np.ndarray:
        """
        Handle outliers in the data
        
        Args:
            X: Feature array
            method: Method for outlier detection ('iqr', 'zscore')
            
        Returns:
            Data with outliers handled
        """
        if not config.HANDLE_OUTLIERS:
            return X
        
        logger.info(f"Handling outliers using {method} method...")
        
        if method == 'iqr':
            # IQR method
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Clip outliers
            X_clipped = np.clip(X, lower_bound, upper_bound)
            
        elif method == 'zscore':
            # Z-score method
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            z_scores = np.abs((X - mean) / (std + 1e-8))
            
            # Clip values with z-score > 3
            X_clipped = X.copy()
            mask = z_scores > 3
            for i in range(X.shape[1]):
                if mask[:, i].any():
                    median = np.median(X[:, i])
                    X_clipped[mask[:, i], i] = median
        else:
            raise ValueError(f"Unknown outlier method: {method}")
        
        n_outliers = np.sum(X != X_clipped)
        logger.info(f"Handled {n_outliers} outlier values")
        
        return X_clipped
    
    def normalize_features(self, X: np.ndarray, method: str = 'standard', 
                          fit: bool = True) -> np.ndarray:
        """
        Normalize features
        
        Args:
            X: Feature array
            method: Normalization method ('standard', 'minmax', 'robust')
            fit: Whether to fit the scaler (True for training, False for test)
            
        Returns:
            Normalized features
        """
        logger.info(f"Normalizing features using {method} method...")
        
        if fit:
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            elif method == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            X_normalized = self.scaler.fit_transform(X)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_normalized = self.scaler.transform(X)
        
        logger.info("Feature normalization complete")
        return X_normalized
    
    def encode_labels(self, y: pd.Series, fit: bool = True) -> np.ndarray:
        """
        Encode categorical labels to numeric
        
        Args:
            y: Label series
            fit: Whether to fit the encoder
            
        Returns:
            Encoded labels
        """
        logger.info("Encoding labels...")
        
        if fit:
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            self.class_names = self.label_encoder.classes_
            logger.info(f"Classes: {self.class_names}")
        else:
            if self.label_encoder is None:
                raise ValueError("Label encoder not fitted. Call with fit=True first.")
            y_encoded = self.label_encoder.transform(y)
        
        return y_encoded
    
    def balance_data(self, X: np.ndarray, y: np.ndarray, 
                    method: str = 'smote') -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance class distribution
        
        Args:
            X: Features
            y: Labels
            method: Balancing method ('smote', 'undersample', 'oversample', 'none')
            
        Returns:
            Balanced X and y
        """
        if method == 'none':
            return X, y
        
        logger.info(f"Balancing data using {method}...")
        logger.info(f"Original class distribution: {np.bincount(y)}")
        
        if method == 'smote':
            # SMOTE oversampling
            smote = SMOTE(random_state=config.RANDOM_STATE)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
        elif method == 'undersample':
            # Random undersampling
            undersampler = RandomUnderSampler(random_state=config.RANDOM_STATE)
            X_balanced, y_balanced = undersampler.fit_resample(X, y)
            
        elif method == 'oversample':
            # Random oversampling
            from imblearn.over_sampling import RandomOverSampler
            oversampler = RandomOverSampler(random_state=config.RANDOM_STATE)
            X_balanced, y_balanced = oversampler.fit_resample(X, y)
        else:
            raise ValueError(f"Unknown balancing method: {method}")
        
        logger.info(f"Balanced class distribution: {np.bincount(y_balanced)}")
        
        return X_balanced, y_balanced
    
    def preprocess(self, data: pd.DataFrame, 
                  binary: bool = False,
                  fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline
        
        Args:
            data: Input DataFrame
            binary: Whether to do binary classification (benign vs malicious)
            fit: Whether to fit preprocessors
            
        Returns:
            Preprocessed X and y
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Handle missing values
        data = self.handle_missing_values(data)
        
        # Find label column
        label_col = None
        for col in data.columns:
            if 'label' in col.lower():
                label_col = col
                break
        
        if label_col is None:
            raise ValueError("Label column not found in data")
        
        # Separate features and labels
        y = data[label_col]
        X = data.drop(columns=[label_col])
        
        # Remove non-numeric columns (if any)
        X = X.select_dtypes(include=[np.number])
        self.feature_names = X.columns.tolist()
        
        # Convert to numpy arrays
        X = X.values
        
        # Handle infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Handle outliers
        if fit:
            X = self.handle_outliers(X, method=config.OUTLIER_METHOD)
        
        # Normalize features
        X = self.normalize_features(X, method=config.NORMALIZATION_METHOD, fit=fit)
        
        # Encode labels
        if binary:
            # Binary classification: benign (0) vs all attacks (1)
            y = y.apply(lambda x: 0 if 'benign' in str(x).lower() else 1)
            y = y.values
            # Set class names for visualization/interpretability
            self.class_names = np.array(['benign', 'malicious'])
        else:
            # Multi-class classification
            y = self.encode_labels(y, fit=fit)
        
        # Balance data (only for training)
        if fit and config.BALANCE_METHOD != 'none':
            X, y = self.balance_data(X, y, method=config.BALANCE_METHOD)
        
        logger.info(f"Preprocessing complete. Final shape: X={X.shape}, y={y.shape}")
        
        return X, y


def main():
    """Main function for standalone preprocessing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess CICIDS2017 dataset')
    parser.add_argument('--explore', action='store_true', help='Perform EDA only')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess and save data')
    parser.add_argument('--output', type=str, default='processed_data.npz', 
                       help='Output file for preprocessed data')
    parser.add_argument('--binary', action='store_true', help='Binary classification')
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor()
    
    try:
        data = preprocessor.load_data()
        
        if args.explore:
            eda_stats = preprocessor.explore_data(data)
            print("\n=== EDA Statistics ===")
            print(f"Samples: {eda_stats['n_samples']}")
            print(f"Features: {eda_stats['n_features']}")
            print(f"\nClass distribution:")
            for class_name, count in eda_stats['class_distribution'].items():
                print(f"  {class_name}: {count}")
        
        if args.preprocess:
            X, y = preprocessor.preprocess(data, binary=args.binary)
            
            # Save preprocessed data
            np.savez(args.output, X=X, y=y, 
                    feature_names=preprocessor.feature_names,
                    class_names=preprocessor.class_names)
            logger.info(f"Saved preprocessed data to {args.output}")
            
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error("Please download the CICIDS2017 dataset and place it in the data/ directory")


if __name__ == '__main__':
    main()
