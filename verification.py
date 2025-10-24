"""
Verification script for Logistic Regression implementation
This script demonstrates Problems 5, 6, 7, and 8
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent segfault on Windows
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from scratch_logistic_regression import ScratchLogisticRegression


def load_and_preprocess_data():
    """
    Load iris dataset and prepare binary classification data (versicolor vs virginica)
    
    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test, X_train_full, X_test_full, feature_names
    """
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    
    # Binary classification: versicolor (1) vs virginica (2)
    # Filter only classes 1 and 2
    mask = (y == 1) | (y == 2)
    X = X[mask]
    y = y[mask]
    
    # Convert labels to 0 and 1 (versicolor=0, virginica=1)
    y = (y == 2).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Also prepare dataset with only 2 features for visualization (Problem 7)
    # Using sepal width (index 1) and petal length (index 2)
    X_train_2d = X_train[:, [1, 2]]
    X_test_2d = X_test[:, [1, 2]]
    
    # Standardize 2D features
    scaler_2d = StandardScaler()
    X_train_2d_scaled = scaler_2d.fit_transform(X_train_2d)
    X_test_2d_scaled = scaler_2d.transform(X_test_2d)
    
    return (X_train_scaled, X_test_scaled, y_train, y_test, 
            X_train_2d_scaled, X_test_2d_scaled, feature_names, scaler)


def train_scratch_model(X_train, y_train, X_val, y_val, verbose=True):
    """
    Train scratch implementation of logistic regression
    
    Parameters
    ----------
    X_train : ndarray
        Training features
    y_train : ndarray
        Training labels
    X_val : ndarray
        Validation features
    y_val : ndarray
        Validation labels
    verbose : bool
        Whether to print training progress
        
    Returns
    -------
    ScratchLogisticRegression
        Trained model
    """
    model = ScratchLogisticRegression(
        num_iter=1000,
        lr=0.1,
        bias=True,
        verbose=verbose
    )
    
    model.fit(X_train, y_train, X_val, y_val, lambda_reg=0.01)
    
    return model


def train_sklearn_model(X_train, y_train):
    """
    Train scikit-learn implementation for comparison
    
    Parameters
    ----------
    X_train : ndarray
        Training features
    y_train : ndarray
        Training labels
        
    Returns
    -------
    LogisticRegression
        Trained sklearn model
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance
    
    Parameters
    ----------
    model : object
        Trained model (scratch or sklearn)
    X_test : ndarray
        Test features
    y_test : ndarray
        Test labels
    model_name : str
        Name of the model for display
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"{model_name} Performance Metrics")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Versicolor', 'Virginica']))
    
    return accuracy, precision, recall


def plot_learning_curve(model, save_path='plots/learning_curve.png'):
    """
    Problem 6: Plot learning curve
    
    Parameters
    ----------
    model : ScratchLogisticRegression
        Trained model
    save_path : str
        Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(model.iter), model.loss, label='Training Loss', linewidth=2)
    plt.plot(range(model.iter), model.val_loss, label='Validation Loss', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Learning Curve: Loss vs Iteration', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"\nLearning curve saved to {save_path}")


def plot_decision_boundary(model, X, y, feature_names, save_path='plots/decision_boundary.png'):
    """
    Problem 7: Visualize decision boundary
    
    Parameters
    ----------
    model : ScratchLogisticRegression
        Trained model (must be trained with 2 features)
    X : ndarray, shape (n_samples, 2)
        Features (2D)
    y : ndarray
        Labels
    feature_names : list
        Names of the two features
    save_path : str
        Path to save the plot
    """
    # Create mesh grid
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict for each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=100, edgecolors='black', 
                         linewidth=1.5, cmap='RdYlBu', alpha=0.8)
    
    plt.xlabel(feature_names[0], fontsize=12)
    plt.ylabel(feature_names[1], fontsize=12)
    plt.title('Decision Boundary Visualization', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Class')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Decision boundary plot saved to {save_path}")


def demonstrate_weight_saving(model):
    """
    Problem 8: Demonstrate weight saving and loading
    
    Parameters
    ----------
    model : ScratchLogisticRegression
        Trained model
    """
    print(f"\n{'='*50}")
    print("Problem 8: Saving and Loading Weights")
    print(f"{'='*50}")
    
    # Save weights using np.savez
    model.save_weights('models/model_weights')
    
    # Save entire model using pickle
    model.save_model('models/model_complete.pkl')
    
    # Load weights
    new_model = ScratchLogisticRegression(num_iter=1000, lr=0.1, bias=True, verbose=False)
    new_model.load_weights('models/model_weights.npz')
    
    print(f"\nOriginal model coefficients: {model.coef_[:3]}... (showing first 3)")
    print(f"Loaded model coefficients:   {new_model.coef_[:3]}... (showing first 3)")
    print(f"Coefficients match: {np.allclose(model.coef_, new_model.coef_)}")
    
    # Load entire model
    loaded_model = ScratchLogisticRegression.load_model('models/model_complete.pkl')
    print(f"\nLoaded complete model coefficients: {loaded_model.coef_[:3]}... (showing first 3)")
    print(f"Complete model match: {np.allclose(model.coef_, loaded_model.coef_)}")


def main():
    """
    Main function to run all verification steps
    """
    print("="*70)
    print(" LOGISTIC REGRESSION VERIFICATION ")
    print("="*70)
    
    # Load and preprocess data
    print("\n[1] Loading and preprocessing iris dataset...")
    (X_train, X_test, y_train, y_test, 
     X_train_2d, X_test_2d, feature_names, scaler) = load_and_preprocess_data()
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Classes: Versicolor (0) vs Virginica (1)")
    
    # Problem 5: Train and compare models
    print(f"\n{'='*70}")
    print("[2] Problem 5: Training Scratch Implementation")
    print(f"{'='*70}")
    
    # Split training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    scratch_model = train_scratch_model(
        X_train_split, y_train_split, X_val_split, y_val_split, verbose=True
    )
    
    # Evaluate scratch model
    scratch_acc, scratch_prec, scratch_rec = evaluate_model(
        scratch_model, X_test, y_test, "Scratch Implementation"
    )
    
    # Train sklearn model for comparison
    print(f"\n{'='*70}")
    print("[3] Training Scikit-learn Implementation for Comparison")
    print(f"{'='*70}")
    sklearn_model = train_sklearn_model(X_train, y_train)
    sklearn_acc, sklearn_prec, sklearn_rec = evaluate_model(
        sklearn_model, X_test, y_test, "Scikit-learn Implementation"
    )
    
    # Compare results
    print(f"\n{'='*70}")
    print("Model Comparison Summary")
    print(f"{'='*70}")
    print(f"{'Metric':<15} {'Scratch':<15} {'Scikit-learn':<15} {'Difference':<15}")
    print(f"{'-'*60}")
    print(f"{'Accuracy':<15} {scratch_acc:<15.4f} {sklearn_acc:<15.4f} {abs(scratch_acc - sklearn_acc):<15.4f}")
    print(f"{'Precision':<15} {scratch_prec:<15.4f} {sklearn_prec:<15.4f} {abs(scratch_prec - sklearn_prec):<15.4f}")
    print(f"{'Recall':<15} {scratch_rec:<15.4f} {sklearn_rec:<15.4f} {abs(scratch_rec - sklearn_rec):<15.4f}")
    
    # Problem 6: Plot learning curve
    print(f"\n{'='*70}")
    print("[4] Problem 6: Plotting Learning Curve")
    print(f"{'='*70}")
    plot_learning_curve(scratch_model)
    
    # Problem 7: Decision boundary visualization
    print(f"\n{'='*70}")
    print("[5] Problem 7: Decision Boundary Visualization")
    print(f"{'='*70}")
    print("Training model with 2 features (Sepal Width and Petal Length) for visualization...")
    
    # Train model with only 2 features
    X_train_2d_split, X_val_2d_split, y_train_2d_split, y_val_2d_split = train_test_split(
        X_train_2d, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    model_2d = train_scratch_model(
        X_train_2d_split, y_train_2d_split, X_val_2d_split, y_val_2d_split, verbose=False
    )
    
    # Combine train and test for visualization
    X_all_2d = np.vstack([X_train_2d, X_test_2d])
    y_all = np.hstack([y_train, y_test])
    
    plot_decision_boundary(
        model_2d, X_all_2d, y_all, 
        ['Sepal Width (standardized)', 'Petal Length (standardized)']
    )
    
    # Problem 8: Save and load weights
    demonstrate_weight_saving(scratch_model)
    
    print(f"\n{'='*70}")
    print(" VERIFICATION COMPLETE ")
    print(f"{'='*70}")
    print("\nAll problems completed successfully!")
    print("\nGenerated files:")
    print("  [Plots]")
    print("     - plots/learning_curve.png")
    print("     - plots/decision_boundary.png")
    print("  [Models]")
    print("     - models/model_weights.npz")
    print("     - models/model_complete.pkl")


if __name__ == "__main__":
    main()

