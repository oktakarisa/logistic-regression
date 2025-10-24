import numpy as np
import pickle


class ScratchLogisticRegression():
    """
    Scratch implementation of logistic regression

    Parameters
    ----------
    num_iter : int
      Number of iterations
    lr : float
      Learning rate
    no_bias : bool
      True if no bias term is included
    verbose : bool
      True to output the learning process

    Attributes
    ----------
    self.coef_ : The following form of ndarray, shape (n_features,)
      Parameters
    self.loss : The following form of ndarray, shape (self.iter,)
      Record losses on training data
    self.val_loss : The following form of ndarray, shape (self.iter,)
      Record loss on validation data

    """

    def __init__(self, num_iter, lr, bias, verbose):
        # Record hyperparameters as attributes
        self.iter = num_iter
        self.lr = lr
        self.bias = bias
        self.verbose = verbose
        # Prepare an array to record the loss
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)
        self.coef_ = None
        
    def _sigmoid(self, z):
        """
        Sigmoid function
        
        Parameters
        ----------
        z : ndarray
            Linear combination of features and weights
            
        Returns
        -------
        ndarray
            Sigmoid output
        """
        # Clip values to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _hypothesis(self, X):
        """
        Hypothesis function for logistic regression
        h_θ(x) = 1 / (1 + e^(-θ^T * x))
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Features
            
        Returns
        -------
        ndarray, shape (n_samples,)
            Predicted probabilities
        """
        z = np.dot(X, self.coef_)
        return self._sigmoid(z)
    
    def _compute_loss(self, X, y, lambda_reg=0.01):
        """
        Compute the objective function (loss function) with regularization
        J(θ) = (1/m) * Σ[-y*log(h_θ(x)) - (1-y)*log(1-h_θ(x))] + (λ/2m) * Σθ_j^2
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Features
        y : ndarray, shape (n_samples,)
            Target values
        lambda_reg : float
            Regularization parameter
            
        Returns
        -------
        float
            Loss value
        """
        m = X.shape[0]
        h = self._hypothesis(X)
        
        # Clip predictions to prevent log(0)
        epsilon = 1e-15
        h = np.clip(h, epsilon, 1 - epsilon)
        
        # Binary cross-entropy loss
        loss = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        
        # Regularization term (excluding bias term if present)
        if self.bias:
            # Don't regularize the bias term (first coefficient)
            reg_term = (lambda_reg / (2 * m)) * np.sum(self.coef_[1:] ** 2)
        else:
            reg_term = (lambda_reg / (2 * m)) * np.sum(self.coef_ ** 2)
        
        return loss + reg_term
    
    def _gradient_descent(self, X, y, lambda_reg=0.01):
        """
        Perform one step of gradient descent
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Features
        y : ndarray, shape (n_samples,)
            Target values
        lambda_reg : float
            Regularization parameter
        """
        m = X.shape[0]
        h = self._hypothesis(X)
        
        # Compute gradient
        gradient = (1/m) * np.dot(X.T, (h - y))
        
        # Add regularization term (excluding bias term)
        if self.bias:
            # Don't regularize the bias term (first coefficient)
            reg_gradient = np.zeros_like(self.coef_)
            reg_gradient[1:] = (lambda_reg / m) * self.coef_[1:]
            gradient += reg_gradient
        else:
            gradient += (lambda_reg / m) * self.coef_
        
        # Update parameters
        self.coef_ -= self.lr * gradient

    def fit(self, X, y, X_val=None, y_val=None, lambda_reg=0.01):
        """
        Learn logistic regression. If validation data is entered, the loss and 
        accuracy for it are also calculated for each iteration.

        Parameters
        ----------
        X : The following forms of ndarray, shape (n_samples, n_features)
            Features of training data
        y : The following form of ndarray, shape (n_samples,)
            Correct answer value of training data
        X_val : The following forms of ndarray, shape (n_samples, n_features)
            Features of verification data
        y_val : The following form of ndarray, shape (n_samples,)
            Correct value of verification data
        lambda_reg : float
            Regularization parameter (default: 0.01)
        """
        # Add bias term if required
        if self.bias:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
            if X_val is not None:
                X_val = np.hstack([np.ones((X_val.shape[0], 1)), X_val])
        
        # Initialize coefficients
        self.coef_ = np.zeros(X.shape[1])
        
        # Training loop
        for i in range(self.iter):
            # Perform gradient descent
            self._gradient_descent(X, y, lambda_reg)
            
            # Calculate and record training loss
            self.loss[i] = self._compute_loss(X, y, lambda_reg)
            
            # Calculate and record validation loss if validation data is provided
            if X_val is not None and y_val is not None:
                self.val_loss[i] = self._compute_loss(X_val, y_val, lambda_reg)
            
            # Output the learning process when verbose is set to True
            if self.verbose and (i % 100 == 0 or i == self.iter - 1):
                if X_val is not None and y_val is not None:
                    print(f"Iteration {i}: Train Loss = {self.loss[i]:.6f}, Val Loss = {self.val_loss[i]:.6f}")
                else:
                    print(f"Iteration {i}: Train Loss = {self.loss[i]:.6f}")

    def predict(self, X):
        """
        Estimate the label using logistic regression.

        Parameters
        ----------
        X : The following forms of ndarray, shape (n_samples, n_features)
            sample

        Returns
        -------
            The following form of ndarray, shape (n_samples,)
            Estimated result by logistic regression
        """
        # Add bias term if required
        if self.bias:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        # Get probabilities
        probabilities = self._hypothesis(X)
        
        # Apply threshold of 0.5
        predictions = (probabilities >= 0.5).astype(int)
        
        return predictions

    def predict_proba(self, X):
        """
        Estimate the probability using logistic regression.

        Parameters
        ----------
        X : The following forms of ndarray, shape (n_samples, n_features)
            sample

        Returns
        -------
            The following form of ndarray, shape (n_samples,)
            Estimated result by logistic regression
        """
        # Add bias term if required
        if self.bias:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        # Return probabilities
        return self._hypothesis(X)
    
    def save_weights(self, filepath):
        """
        Save the learned weights to a file
        
        Parameters
        ----------
        filepath : str
            Path to save the weights
        """
        np.savez(filepath, coef=self.coef_, loss=self.loss, val_loss=self.val_loss)
        print(f"Weights saved to {filepath}.npz")
    
    def load_weights(self, filepath):
        """
        Load weights from a file
        
        Parameters
        ----------
        filepath : str
            Path to load the weights from
        """
        data = np.load(filepath)
        self.coef_ = data['coef']
        self.loss = data['loss']
        self.val_loss = data['val_loss']
        print(f"Weights loaded from {filepath}")
    
    def save_model(self, filepath):
        """
        Save the entire model using pickle
        
        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """
        Load a model from a file
        
        Parameters
        ----------
        filepath : str
            Path to load the model from
            
        Returns
        -------
        ScratchLogisticRegression
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model

