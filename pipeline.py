import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, 
                           precision_score, recall_score, auc)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, fallback if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("XGBoost is available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, using sklearn models only")

class FraudDetectionPipeline:
    def __init__(self, data_path=None):
        """
        Initialize the Fraud Detection Pipeline
        
        Args:
            data_path (str): Path to the credit card dataset CSV file
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_resampled = None
        self.y_resampled = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_and_explore_data(self, sample_data=True):
        """
        Load and explore the credit card fraud dataset
        
        Args:
            sample_data (bool): If True, creates sample data for demonstration
        """
        if sample_data or self.data_path is None:
            print("Creating sample dataset for demonstration...")
            # Create a sample dataset with similar characteristics
            np.random.seed(42)
            n_samples = 10000
            n_features = 30
            
            # Generate normal transactions (99.83%)
            n_normal = int(n_samples * 0.9983)
            normal_data = np.random.normal(0, 1, (n_normal, n_features))
            normal_labels = np.zeros(n_normal)
            
            # Generate fraudulent transactions (0.17%)
            n_fraud = n_samples - n_normal
            # Fraudulent transactions have different patterns
            fraud_data = np.random.normal(2, 1.5, (n_fraud, n_features))
            fraud_labels = np.ones(n_fraud)
            
            # Combine data
            X = np.vstack([normal_data, fraud_data])
            y = np.hstack([normal_labels, fraud_labels])
            
            # Create feature names similar to the real dataset
            feature_names = [f'V{i}' for i in range(1, n_features)] + ['Amount']
            
            # Create DataFrame
            self.df = pd.DataFrame(X, columns=feature_names)
            self.df['Class'] = y.astype(int)
            
            # Add some realistic patterns to Amount feature
            self.df.loc[self.df['Class'] == 0, 'Amount'] = np.random.lognormal(3, 1, n_normal)
            self.df.loc[self.df['Class'] == 1, 'Amount'] = np.random.lognormal(5, 1.5, n_fraud)
            
        else:
            print(f"Loading dataset from {self.data_path}...")
            self.df = pd.read_csv(self.data_path)
        
        print("Dataset loaded successfully!")
        self.explore_dataset()
        
    def explore_dataset(self):
        """Explore the dataset characteristics"""
        print("\n" + "="*50)
        print("DATASET EXPLORATION")
        print("="*50)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Features: {list(self.df.columns)}")
        
        # Class distribution
        print("\nClass Distribution:")
        class_counts = self.df['Class'].value_counts()
        class_props = self.df['Class'].value_counts(normalize=True)
        
        print(f"Normal transactions (0): {class_counts[0]} ({class_props[0]:.4f})")
        print(f"Fraudulent transactions (1): {class_counts[1]} ({class_props[1]:.4f})")
        
        # Check for missing values
        print(f"\nMissing values: {self.df.isnull().sum().sum()}")
        
        # Basic statistics
        print("\nDataset Info:")
        print(self.df.describe())
        
        # Visualizations
        self.plot_class_distribution()
        self.plot_feature_distributions()
        
    def plot_class_distribution(self):
        """Plot class distribution"""
        plt.figure(figsize=(12, 4))
        
        # Count plot
        plt.subplot(1, 2, 1)
        self.df['Class'].value_counts().plot(kind='bar')
        plt.title('Class Distribution (Count)')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['Normal', 'Fraud'], rotation=0)
        
        # Proportion plot
        plt.subplot(1, 2, 2)
        self.df['Class'].value_counts(normalize=True).plot(kind='bar')
        plt.title('Class Distribution (Proportion)')
        plt.xlabel('Class')
        plt.ylabel('Proportion')
        plt.xticks([0, 1], ['Normal', 'Fraud'], rotation=0)
        
        plt.tight_layout()
        plt.show()
        
    def plot_feature_distributions(self):
        """Plot distributions of key features"""
        plt.figure(figsize=(15, 10))
        
        # Plot Amount distribution
        plt.subplot(2, 2, 1)
        plt.hist(self.df[self.df['Class'] == 0]['Amount'], bins=50, alpha=0.7, label='Normal', density=True)
        plt.hist(self.df[self.df['Class'] == 1]['Amount'], bins=50, alpha=0.7, label='Fraud', density=True)
        plt.xlabel('Amount')
        plt.ylabel('Density')
        plt.title('Amount Distribution by Class')
        plt.legend()
        plt.yscale('log')
        
        # Plot first few V features
        features_to_plot = [col for col in self.df.columns if col.startswith('V')][:3]
        
        for i, feature in enumerate(features_to_plot):
            plt.subplot(2, 2, i+2)
            plt.hist(self.df[self.df['Class'] == 0][feature], bins=50, alpha=0.7, label='Normal', density=True)
            plt.hist(self.df[self.df['Class'] == 1][feature], bins=50, alpha=0.7, label='Fraud', density=True)
            plt.xlabel(feature)
            plt.ylabel('Density')
            plt.title(f'{feature} Distribution by Class')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def preprocess_data(self):
        """Preprocess the data and create train-test split"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Separate features and target
        X = self.df.drop('Class', axis=1)
        y = self.df['Class']
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Stratified train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        # Check class distribution in splits
        print("\nClass distribution in training set:")
        print(self.y_train.value_counts(normalize=True))
        
        print("\nClass distribution in test set:")
        print(self.y_test.value_counts(normalize=True))
        
        # Feature scaling
        print("\nApplying feature scaling...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Preprocessing completed!")
        
    def handle_imbalance(self, method='smote'):
        """
        Handle class imbalance using various techniques
        
        Args:
            method (str): Method to handle imbalance ('smote', 'undersampling', 'smotetomek')
        """
        print("\n" + "="*50)
        print("HANDLING CLASS IMBALANCE")
        print("="*50)
        
        print(f"Original training set distribution:")
        print(f"Normal: {sum(self.y_train == 0)} ({sum(self.y_train == 0)/len(self.y_train):.4f})")
        print(f"Fraud: {sum(self.y_train == 1)} ({sum(self.y_train == 1)/len(self.y_train):.4f})")
        
        if method == 'smote':
            print("\nApplying SMOTE...")
            smote = SMOTE(random_state=42)
            self.X_resampled, self.y_resampled = smote.fit_resample(self.X_train_scaled, self.y_train)
            
        elif method == 'undersampling':
            print("\nApplying Random Under Sampling...")
            undersampler = RandomUnderSampler(random_state=42)
            self.X_resampled, self.y_resampled = undersampler.fit_resample(self.X_train_scaled, self.y_train)
            
        elif method == 'smotetomek':
            print("\nApplying SMOTE + Tomek...")
            smotetomek = SMOTETomek(random_state=42)
            self.X_resampled, self.y_resampled = smotetomek.fit_resample(self.X_train_scaled, self.y_train)
        
        print(f"\nResampled training set distribution:")
        print(f"Normal: {sum(self.y_resampled == 0)} ({sum(self.y_resampled == 0)/len(self.y_resampled):.4f})")
        print(f"Fraud: {sum(self.y_resampled == 1)} ({sum(self.y_resampled == 1)/len(self.y_resampled):.4f})")
        
        print(f"Resampled training set shape: {self.X_resampled.shape}")
        
    def train_models(self):
        """Train multiple models for comparison"""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Random Forest (Balanced)': RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            models['XGBoost (Balanced)'] = xgb.XGBClassifier(
                random_state=42, eval_metric='logloss',
                scale_pos_weight=sum(self.y_train == 0) / sum(self.y_train == 1)
            )
        
        # Train models
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            if 'Balanced' in name and self.X_resampled is not None:
                # Use original imbalanced data for balanced models
                model.fit(self.X_train_scaled, self.y_train)
            else:
                # Use resampled data for other models
                if self.X_resampled is not None:
                    model.fit(self.X_resampled, self.y_resampled)
                else:
                    model.fit(self.X_train_scaled, self.y_train)
            
            self.models[name] = model
            print(f"{name} trained successfully!")
        
        print(f"\nTrained {len(self.models)} models successfully!")
        
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        for name, model in self.models.items():
            print(f"\n{'='*20} {name} {'='*20}")
            
            # Predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate metrics
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Precision-Recall AUC
            precision_curve, recall_curve, _ = precision_recall_curve(self.y_test, y_pred_proba)
            pr_auc = auc(recall_curve, precision_curve)
            
            # Store results
            self.results[name] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc
            }
            
            # Print metrics
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
            print(f"PR AUC: {pr_auc:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            print(f"\nConfusion Matrix:")
            print(f"True Negative: {cm[0,0]}, False Positive: {cm[0,1]}")
            print(f"False Negative: {cm[1,0]}, True Positive: {cm[1,1]}")
            
            # Classification Report
            print(f"\nClassification Report:")
            print(classification_report(self.y_test, y_pred))
        
        # Plot comparison
        self.plot_model_comparison()
        
    def plot_model_comparison(self):
        """Plot model comparison charts"""
        if not self.results:
            print("No results to plot. Run evaluate_models() first.")
            return
            
        # Metrics comparison
        plt.figure(figsize=(15, 12))
        
        # ROC Curves
        plt.subplot(2, 3, 1)
        for name, results in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
            plt.plot(fpr, tpr, label=f"{name} (AUC = {results['roc_auc']:.3f})")
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Precision-Recall Curves
        plt.subplot(2, 3, 2)
        for name, results in self.results.items():
            precision, recall, _ = precision_recall_curve(self.y_test, results['y_pred_proba'])
            plt.plot(recall, precision, label=f"{name} (PR AUC = {results['pr_auc']:.3f})")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Metrics comparison bar plot
        metrics = ['precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        
        start_index = 3  # Start from subplot 3
        for i, metric in enumerate(metrics):
            if i + start_index > 6:
                plt.figure(figsize=(10, 5))  # Optional: create new figure if overflow
                start_index = 1
            plt.subplot(2, 3, i + start_index)
            model_names = list(self.results.keys())
            values = [self.results[name][metric] for name in model_names]
            
            bars = plt.bar(range(len(model_names)), values)
            plt.xlabel('Models')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.title(f'{metric.replace("_", " ").title()} Comparison')
            plt.xticks(range(len(model_names)), [name.split()[0] for name in model_names], rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.ylim(0, 1.1)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        if not self.results:
            print("No results to plot. Run evaluate_models() first.")
            return
            
        n_models = len(self.results)
        plt.figure(figsize=(5*n_models, 4))
        
        for i, (name, results) in enumerate(self.results.items()):
            plt.subplot(1, n_models, i+1)
            
            cm = confusion_matrix(self.y_test, results['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Normal', 'Fraud'],
                       yticklabels=['Normal', 'Fraud'])
            plt.title(f'{name}\nConfusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
        
    def anomaly_detection_models(self):
        """Train and evaluate anomaly detection models"""
        print("\n" + "="*50)
        print("ANOMALY DETECTION MODELS")
        print("="*50)
        
        # Train only on normal transactions
        X_normal = self.X_train_scaled[self.y_train == 0]
        
        # Isolation Forest
        print("Training Isolation Forest...")
        iso_forest = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
        iso_forest.fit(X_normal)
        
        # One-Class SVM
        print("Training One-Class SVM...")
        oc_svm = OneClassSVM(gamma='scale', nu=0.1)
        oc_svm.fit(X_normal)
        
        # Evaluate anomaly detection models
        anomaly_models = {
            'Isolation Forest': iso_forest,
            'One-Class SVM': oc_svm
        }
        
        print("\nEvaluating Anomaly Detection Models:")
        
        for name, model in anomaly_models.items():
            # Predict (-1 for anomaly, 1 for normal)
            predictions = model.predict(self.X_test_scaled)
            # Convert to binary (0 for normal, 1 for anomaly)
            y_pred_anomaly = (predictions == -1).astype(int)
            
            # Calculate metrics
            precision = precision_score(self.y_test, y_pred_anomaly)
            recall = recall_score(self.y_test, y_pred_anomaly)
            f1 = f1_score(self.y_test, y_pred_anomaly)
            
            print(f"\n{name}:")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred_anomaly)
            print(f"Confusion Matrix:")
            print(f"True Negative: {cm[0,0]}, False Positive: {cm[0,1]}")
            print(f"False Negative: {cm[1,0]}, True Positive: {cm[1,1]}")
        
    def get_best_model(self):
        """Get the best performing model based on F1 score"""
        if not self.results:
            print("No results available. Run evaluate_models() first.")
            return None
            
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['f1'])
        best_model = self.models[best_model_name]
        best_f1 = self.results[best_model_name]['f1']
        
        print(f"\nBest performing model: {best_model_name}")
        print(f"F1-Score: {best_f1:.4f}")
        
        return best_model_name, best_model
        
    def feature_importance_analysis(self):
        """Analyze feature importance for tree-based models"""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        # Get feature names
        feature_names = self.df.drop('Class', axis=1).columns
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                print(f"\n{name} Feature Importance:")
                
                # Get feature importances
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Print top 10 features
                print("Top 10 Most Important Features:")
                for i in range(min(10, len(feature_names))):
                    idx = indices[i]
                    print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
                
                # Plot feature importance
                plt.figure(figsize=(12, 6))
                plt.title(f"Feature Importance - {name}")
                plt.bar(range(min(20, len(feature_names))), 
                        importances[indices[:min(20, len(feature_names))]])
                plt.xticks(range(min(20, len(feature_names))), 
                          [feature_names[i] for i in indices[:min(20, len(feature_names))]], 
                          rotation=45)
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.tight_layout()
                plt.show()
                
                break  # Show only for first tree-based model
                
    def run_complete_pipeline(self, resampling_method='smote'):
        """Run the complete fraud detection pipeline"""
        print("🚀 Starting Credit Card Fraud Detection Pipeline")
        print("="*60)
        
        # Step 1: Load and explore data
        self.load_and_explore_data()
        
        # Step 2: Preprocess data
        self.preprocess_data()
        
        # Step 3: Handle imbalance
        self.handle_imbalance(method=resampling_method)
        
        # Step 4: Train models
        self.train_models()
        
        # Step 5: Evaluate models
        self.evaluate_models()
        
        # Step 6: Plot confusion matrices
        self.plot_confusion_matrices()
        
        # Step 7: Feature importance analysis
        self.feature_importance_analysis()
        
        # Step 8: Anomaly detection models
        self.anomaly_detection_models()
        
        # Step 9: Get best model
        best_model_info = self.get_best_model()
        
        print("\n🎉 Pipeline completed successfully!")
        return best_model_info

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = FraudDetectionPipeline(data_path="creditcard.csv")
    
    # Run complete pipeline
    best_model = pipeline.run_complete_pipeline(resampling_method='smote')
    
    # Print summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print("Dataset loaded and explored")
    print("Data preprocessed and split")
    print("Class imbalance handled with SMOTE")
    print("Multiple models trained and evaluated")
    print("Performance metrics calculated")
    print("Visualizations created")
    print("Feature importance analyzed")
    print("Anomaly detection models evaluated")
    
    print(f"\n Best Model: {best_model[0] if best_model else 'N/A'}")
    
    print("\n Key Insights for Fraud Detection:")
    print("• Recall is crucial - we want to catch fraudulent transactions")
    print("• Precision matters - avoid flagging too many legitimate transactions")
    print("• F1-score provides good balance between precision and recall")
    print("• PR-AUC is better than ROC-AUC for imbalanced datasets")
    print("• Ensemble methods often perform better")
    print("• Feature importance helps understand fraud patterns")