# Problem Statement:
You are required to implement a robust binary and multi class intrusion detection system capable of identifying malicious and benign traffic flows.
The model should be trained using the **CICIDS2017** dataset.
You will implement Logistic Regression and SVM, tune their hyperparameters and analyze their performance, interpretability and resistance to adversarial noise.
### Dataset: CICIDS2017

# Technical Requirements:
1. Implement both SVM and Logistic Regression from scratch, including optimization routines (gradient descent, hinge loss, regularization).
2. Perform detailed EDA and preprocessing: normalization, outlier handling, feature encoding and data balancing.
3. Include gradient checking to validate your optimization 
implementation.
4. Use appropriate kernel functions (linear, polynomial, RBF) for SVM and compare their effects.
5. Include interpretability analysis using SHAP or LIME to explain model predictions.
6. Evaluate on both clean and noisy/adversarially perturbed data.
7. Perform hyperparameter tuning and document your methodology.
8. Visualize results (decision boundaries, confusion matrices, ROC curves, etc.).

# Bonus Tasks (Optional)
1. Implement stochastic gradient descent (SGD) and mini batch training for both models.
2. Integrate a simple adversarial attack generator (FGSM or random noise) to test model robustness.
3. Explore model compression or pruning for lightweight deployment.
4. Train the models on subsets of CICIDS2017 and compare specialization effects. 


# Deliverables:
1. GitHub repository containing:
- Source code for both models (no prebuilt ML libraries).
- A README explaining setup, dataset usage and results.
- Visualization and interpretability scripts.
2. A report containing:
- Problem description, methodology, analysis, results and conclusion.
- Screenshots of results and visualizations.
- Training logs and performance metrics.
- Link to Github repository.


# Evaluation Criteria:
- Correctness and completeness of SVM and Logistic Regression implementations.
- Quality of EDA, preprocessing and data understanding.
- Interpretability and clarity of analysis (use of SHAP/LIME).
- Robustness testing and adversarial evaluation.
- Depth of explanation in written report.
- Commit history and organization in private GitHub repository. Overall code quality, readability and reproducibility