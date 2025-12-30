"""
SVM Classifier Module
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


class SVMClassifier:
    """SVM classifier for cat meow recognition"""
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', random_state=42):
        """Initialize SVM classifier"""
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=random_state,
            probability=True
        )
        self.is_fitted = False
    
    def fit(self, X, y):
        """Train the model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict class labels"""
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X, y, target_names=None):
        """Evaluate model performance"""
        y_pred = self.predict(X)
        
        results = {
            'accuracy': accuracy_score(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'classification_report': classification_report(
                y, y_pred, target_names=target_names, output_dict=True
            )
        }
        
        return results
    
    def cross_validate(self, X, y, cv=5):
        """Cross validation (using Pipeline to avoid data leakage)"""
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import StratifiedKFold
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel=self.kernel, C=self.C, gamma=self.gamma,
                       random_state=self.random_state, probability=True))
        ])
        
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y, cv=cv_strategy)
        
        return {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'scores': scores
        }
    
    def grid_search(self, X, y, param_grid=None, cv=5):
        """Grid search for optimal parameters"""
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'kernel': ['rbf', 'linear', 'poly']
            }
        
        X_scaled = self.scaler.fit_transform(X)
        
        grid_search = GridSearchCV(
            SVC(random_state=self.random_state, probability=True),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_scaled, y)
        
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        return self, {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def save(self, path):
        """Save model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }, path)
    
    def load(self, path):
        """Load model"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_fitted = data['is_fitted']
        return self
