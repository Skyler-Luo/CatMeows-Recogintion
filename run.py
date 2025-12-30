"""
Cat Meow Recognition - 10-fold Cross Validation (SVM + MFCC)
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             f1_score, precision_score, recall_score, roc_auc_score)
from datetime import datetime

from src.data.loader import load_dataset, CATEGORY_NAMES
from src.features.mfcc import extract_mfcc_features
from src.models.svm import SVMClassifier
from tools.visualization import (visualize_all_features, plot_confusion_matrix,
                                  plot_roc_curve, plot_pr_curve)
from tools.logger import setup_logger, TrainingResultSaver


def cross_validate_svm(features, labels, kernel='rbf', C=10, gamma='scale', cv=10, logger=None):
    """SVM 10-fold cross validation"""
    log = logger.info if logger else print
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        svm = SVMClassifier(kernel=kernel, C=C, gamma=gamma)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        y_proba = svm.predict_proba(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)
    
    target_names = [CATEGORY_NAMES[i] for i in range(len(CATEGORY_NAMES))]
    n_classes = len(target_names)
    
    # Calculate metrics
    results = {
        'accuracy': np.mean(scores),
        'std_accuracy': np.std(scores),
        'cv_scores': scores,
        'f1_macro': f1_score(all_y_true, all_y_pred, average='macro'),
        'f1_weighted': f1_score(all_y_true, all_y_pred, average='weighted'),
        'precision_macro': precision_score(all_y_true, all_y_pred, average='macro'),
        'recall_macro': recall_score(all_y_true, all_y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(all_y_true, all_y_pred),
        'classification_report': classification_report(
            all_y_true, all_y_pred, target_names=target_names, output_dict=True
        ),
        'feature_dim': features.shape[1],
        'y_true': all_y_true,
        'y_pred': all_y_pred,
        'y_proba': all_y_proba
    }
    
    # Calculate AUC (one-vs-rest for multiclass)
    try:
        results['auc_macro'] = roc_auc_score(all_y_true, all_y_proba, multi_class='ovr', average='macro')
        results['auc_weighted'] = roc_auc_score(all_y_true, all_y_proba, multi_class='ovr', average='weighted')
    except Exception:
        results['auc_macro'] = None
        results['auc_weighted'] = None
    
    log(f"10-fold CV accuracy: {results['accuracy']:.4f} (+/-{results['std_accuracy']:.4f})")
    log(f"F1 (macro): {results['f1_macro']:.4f}")
    log(f"Precision (macro): {results['precision_macro']:.4f}")
    log(f"Recall (macro): {results['recall_macro']:.4f}")
    if results['auc_macro']:
        log(f"AUC (macro): {results['auc_macro']:.4f}")
    
    return results


def main():
    """Main function"""
    saver = TrainingResultSaver("outputs")
    logger = setup_logger("cat_recognition", saver.get_run_dir())
    
    logger.info("=" * 50)
    logger.info("Cat Meow Recognition System (SVM + MFCC)")
    logger.info("=" * 50)
    logger.info(f"Output directory: {saver.get_run_dir()}")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("dataset")
    audio_data, labels = dataset.load_all_audio()
    logger.info(f"Loaded {len(audio_data)} samples")
    
    target_names = [CATEGORY_NAMES[i] for i in range(len(CATEGORY_NAMES))]
    
    # Extract MFCC features
    logger.info("Extracting MFCC features...")
    mfcc_features = extract_mfcc_features(
        audio_data, n_mfcc=20, n_mels=26, pooling='stats',
        include_delta=True, include_delta2=True, include_delta3=False
    )
    mfcc_features = np.nan_to_num(mfcc_features, nan=0.0, posinf=0.0, neginf=0.0)
    logger.info(f"Feature dimension: {mfcc_features.shape}")
    
    # 10-fold cross validation
    logger.info("Starting 10-fold cross validation...")
    results = cross_validate_svm(mfcc_features, labels, logger=logger)
    # Save results (exclude large arrays for JSON)
    results_to_save = {k: v for k, v in results.items() if k not in ['y_true', 'y_pred', 'y_proba']}
    saver.save_results(results_to_save, 'svm_mfcc', logger)
    
    # Train final model
    logger.info("Training final model...")
    final_svm = SVMClassifier(kernel='rbf', C=10, gamma='scale')
    final_svm.fit(mfcc_features, labels)
    
    # Visualizations (including SVM decision boundary)
    logger.info("Generating visualizations...")
    visualize_all_features(
        mfcc_features, labels, target_names,
        output_dir=saver.get_viz_dir(),
        audio_data=audio_data,
        sr=22050,
        svm_model=final_svm
    )
    
    # Confusion matrix
    plot_confusion_matrix(
        results['confusion_matrix'],
        target_names,
        title="Confusion Matrix (SVM + MFCC)",
        save_path=f"{saver.get_viz_dir()}/confusion_matrix.png"
    )
    
    # ROC curve
    plot_roc_curve(
        results['y_true'], results['y_proba'],
        target_names,
        save_path=f"{saver.get_viz_dir()}/roc_curve.png"
    )
    
    # PR curve
    plot_pr_curve(
        results['y_true'], results['y_proba'],
        target_names,
        save_path=f"{saver.get_viz_dir()}/pr_curve.png"
    )
    
    # Save model
    model_path = f"{saver.get_run_dir()}/best_model.pkl"
    final_svm.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset_size': len(audio_data),
        'num_classes': len(CATEGORY_NAMES),
        'class_names': target_names,
        'cv_folds': 10,
        'accuracy': results['accuracy'],
        'std_accuracy': results['std_accuracy'],
        'f1_macro': results['f1_macro'],
        'precision_macro': results['precision_macro'],
        'recall_macro': results['recall_macro'],
        'auc_macro': results['auc_macro']
    }
    saver.save_summary(summary, logger)
    
    logger.info("=" * 50)
    logger.info(f"Accuracy: {results['accuracy']:.4f} (+/-{results['std_accuracy']:.4f})")
    logger.info(f"F1 (macro): {results['f1_macro']:.4f}")
    logger.info(f"AUC (macro): {results['auc_macro']:.4f}" if results['auc_macro'] else "AUC: N/A")
    logger.info(f"Training complete! Output: {saver.get_run_dir()}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
