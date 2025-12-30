"""
Visualization Tools Module
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend, no popup windows
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os


def plot_waveform(y, sr, title="Waveform", save_path=None):
    """Plot audio waveform"""
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_spectrogram(y, sr, title="Spectrogram", save_path=None):
    """Plot spectrogram"""
    plt.figure(figsize=(12, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_mel_spectrogram(y, sr, title="Mel Spectrogram", save_path=None):
    """Plot mel spectrogram"""
    plt.figure(figsize=(12, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_mfcc(y, sr, n_mfcc=13, title="MFCC", save_path=None):
    """Plot MFCC coefficients"""
    plt.figure(figsize=(12, 4))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title(title)
    plt.ylabel('MFCC Coefficients')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curve(y_true, y_proba, class_names, title="ROC Curve", save_path=None):
    """
    Plot ROC curve for multiclass classification (One-vs-Rest)
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities (n_samples, n_classes)
        class_names: Class names
        save_path: Save path
    """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(10, 8))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    
    # Plot ROC curve for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pr_curve(y_true, y_proba, class_names, title="Precision-Recall Curve", save_path=None):
    """
    Plot Precision-Recall curve for multiclass classification (One-vs-Rest)
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities (n_samples, n_classes)
        class_names: Class names
        save_path: Save path
    """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(10, 8))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    
    # Plot PR curve for each class
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_proba[:, i])
        plt.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                 label=f'{class_names[i]} (AP = {ap:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_feature_distribution(features, labels, feature_names, class_names, n_features=6, save_path=None):
    """Plot feature distribution by class"""
    n_features = min(n_features, features.shape[1])
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i in range(n_features):
        ax = axes[i]
        for label in np.unique(labels):
            mask = labels == label
            ax.hist(features[mask, i], bins=20, alpha=0.5, label=class_names[label])
        ax.set_title(feature_names[i] if i < len(feature_names) else f'Feature {i}')
        ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_audio_comparison(audio_list, sr, titles, save_path=None):
    """Plot audio comparison (waveform and spectrogram)"""
    n = len(audio_list)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3*n))
    for i, (y, title) in enumerate(zip(audio_list, titles)):
        librosa.display.waveshow(y, sr=sr, ax=axes[i, 0])
        axes[i, 0].set_title(f'{title} - Waveform')
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[i, 1])
        axes[i, 1].set_title(f'{title} - Spectrogram')
        fig.colorbar(img, ax=axes[i, 1], format='%+2.0f dB')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_feature_heatmap(features, labels, class_names, feature_names=None, title="Feature Heatmap", save_path=None):
    """Plot feature heatmap by class"""
    from sklearn.preprocessing import StandardScaler
    unique_labels = np.unique(labels)
    mean_features = []
    for label in unique_labels:
        mask = labels == label
        mean_features.append(np.mean(features[mask], axis=0))
    mean_features = np.array(mean_features)
    scaler = StandardScaler()
    mean_features_scaled = scaler.fit_transform(mean_features.T).T
    plt.figure(figsize=(14, 6))
    if feature_names is None:
        feature_names = [f'F{i}' for i in range(features.shape[1])]
    max_features = 50
    if len(feature_names) > max_features:
        feature_names = feature_names[:max_features]
        mean_features_scaled = mean_features_scaled[:, :max_features]
    sns.heatmap(mean_features_scaled, xticklabels=feature_names,
                yticklabels=[class_names[i] for i in unique_labels],
                cmap='RdBu_r', center=0, annot=False)
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_feature_pca(features, labels, class_names, n_components=2, title="PCA Feature Visualization", save_path=None):
    """Plot PCA visualization of features"""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)
    if n_components == 2:
        plt.figure(figsize=(10, 8))
        for label in np.unique(labels):
            mask = labels == label
            plt.scatter(features_pca[mask, 0], features_pca[mask, 1],
                       label=class_names[label], alpha=0.7, s=60)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    else:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for label in np.unique(labels):
            mask = labels == label
            ax.scatter(features_pca[mask, 0], features_pca[mask, 1], features_pca[mask, 2],
                      label=class_names[label], alpha=0.7, s=60)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
    return pca


def plot_feature_tsne(features, labels, class_names, perplexity=30, title="t-SNE Feature Visualization", save_path=None):
    """Plot t-SNE visualization of features"""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    features_tsne = tsne.fit_transform(features_scaled)
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                   label=class_names[label], alpha=0.7, s=60)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_feature_correlation(features, feature_names=None, title="Feature Correlation Matrix", save_path=None):
    """Plot feature correlation matrix"""
    corr_matrix = np.corrcoef(features.T)
    max_features = 30
    if corr_matrix.shape[0] > max_features:
        corr_matrix = corr_matrix[:max_features, :max_features]
        if feature_names:
            feature_names = feature_names[:max_features]
    plt.figure(figsize=(12, 10))
    if feature_names is None:
        feature_names = [f'F{i}' for i in range(corr_matrix.shape[0])]
    sns.heatmap(corr_matrix, xticklabels=feature_names, yticklabels=feature_names,
                cmap='coolwarm', center=0, vmin=-1, vmax=1, square=True)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_class_feature_comparison(features, labels, class_names, feature_indices=None,
                                   feature_names=None, title="Feature Comparison by Class", save_path=None):
    """Plot boxplot comparison of features by class"""
    if feature_indices is None:
        feature_indices = list(range(min(6, features.shape[1])))
    n_features = len(feature_indices)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_features == 1 else axes
    for i, feat_idx in enumerate(feature_indices):
        ax = axes[i]
        data = []
        for label in np.unique(labels):
            mask = labels == label
            data.append(features[mask, feat_idx])
        ax.boxplot(data, labels=[class_names[l] for l in np.unique(labels)])
        if feature_names and feat_idx < len(feature_names):
            ax.set_title(feature_names[feat_idx])
        else:
            ax.set_title(f'Feature {feat_idx}')
        ax.set_ylabel('Value')
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_spectrogram_grid(audio_list, sr, labels, class_names, samples_per_class=2,
                          title="Spectrogram Comparison by Class", save_path=None):
    """Plot spectrogram grid for each class"""
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    fig, axes = plt.subplots(n_classes, samples_per_class, figsize=(5*samples_per_class, 3*n_classes))
    if n_classes == 1:
        axes = axes.reshape(1, -1)
    rng = np.random.RandomState(42)
    for i, label in enumerate(unique_labels):
        mask = np.where(labels == label)[0]
        selected = rng.choice(mask, size=min(samples_per_class, len(mask)), replace=False)
        for j, idx in enumerate(selected):
            ax = axes[i, j]
            y = audio_list[idx]
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='jet')
            ax.set_ylim(0, 4000)  # Limit to 0-4000Hz (consistent with MFCC range)
            ax.set_title(f'{class_names[label]} - Sample {j+1}')
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_svm_decision_boundary(features, labels, class_names, svm_model=None,
                                title="SVM Decision Boundary (PCA 2D)", save_path=None):
    """
    Visualize SVM decision boundary (using PCA to reduce to 2D)
    
    Args:
        features: Feature matrix
        labels: Labels
        class_names: Class names
        svm_model: Trained SVM model, None to create a new one
        save_path: Save path
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    
    # Standardize and reduce to 2D with PCA
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_scaled)
    
    # Train SVM in 2D space
    if svm_model is None:
        svm_2d = SVC(kernel='rbf', C=10, gamma='scale')
    else:
        # Use same parameters
        svm_2d = SVC(kernel=svm_model.kernel, C=svm_model.C, gamma=svm_model.gamma)
    svm_2d.fit(features_2d, labels)
    
    # Create mesh grid
    x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
    y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Predict on mesh grid
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Decision regions
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set3')
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5)
    
    # Data points
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    for i, label in enumerate(np.unique(labels)):
        mask = labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=colors[i % len(colors)], label=class_names[label],
                   edgecolors='k', s=60, alpha=0.8)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_all_features(features, labels, class_names, feature_names=None,
                           output_dir="visualizations", audio_data=None, sr=22050,
                           svm_model=None):
    """Generate all visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    print("Generating visualizations...")
    if audio_data is not None:
        print("  - Spectrogram comparison")
        plot_spectrogram_grid(audio_data, sr, labels, class_names, samples_per_class=2,
                              save_path=os.path.join(output_dir, "spectrogram_comparison.png"))
    print("  - t-SNE visualization")
    plot_feature_tsne(features, labels, class_names,
                      save_path=os.path.join(output_dir, "tsne.png"))
    print("  - SVM decision boundary")
    plot_svm_decision_boundary(features, labels, class_names, svm_model,
                               save_path=os.path.join(output_dir, "svm_decision_boundary.png"))
    print(f"Visualizations saved to {output_dir}/")
