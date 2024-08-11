import torch
import numpy as np
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, learning_curve,
                                     train_test_split)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, auc, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.utils import parallel_backend
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import random

def get_confidence_scores(dataset, model, tokenizer):
    model.eval()
    confidence_scores = []
    for batch in dataset:
        with torch.no_grad():
            inputs = tokenizer(batch['text'], return_tensors='pt', truncation=True, max_length=512).to(model.device)
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            probs = torch.softmax(logits, dim=-1)
            confidence_score = torch.max(probs, dim=-1)[0].mean()
            confidence_scores.append(confidence_score)

    return torch.stack(confidence_scores)

def perform_error_analysis(X_test, y_test, y_pred, feature_names):
    misclassified = X_test[y_test != y_pred]
    misclassified_true = y_test[y_test != y_pred]
    misclassified_pred = y_pred[y_test != y_pred]

    print("Error Analysis:")
    print(f"Total misclassified samples: {len(misclassified)}")

    print("\nMisclassification breakdown:")
    print(f"False Positives: {sum((misclassified_true == 0) & (misclassified_pred == 1))}")
    print(f"False Negatives: {sum((misclassified_true == 1) & (misclassified_pred == 0))}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def find_optimal_threshold(y_true, y_pred_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    return optimal_threshold

def visualize_features(X, y, title):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], label='Non-member', alpha=0.6)
    plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], label='Member', alpha=0.6)
    plt.title(f'PCA visualization of {title}')
    plt.legend()
    plt.savefig(f'{title}_pca_visualization.png')
    plt.close()

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], label='Non-member', alpha=0.6)
    plt.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], label='Member', alpha=0.6)
    plt.title(f't-SNE visualization of {title}')
    plt.legend()
    plt.savefig(f'{title}_tsne_visualization.png')
    plt.close()

def plot_feature_importance(feature_importance, feature_names, title):
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    plt.figure(figsize=(12, 8))
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, np.array(feature_names)[sorted_idx])
    plt.title(title)
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(.1, 1.0, 5), scoring="roc_auc")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

class ConfidencePenaltyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, penalty_weight=0.1):
        self.penalty_weight = penalty_weight

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        confidence = np.max(X, axis=1)
        penalty = self.penalty_weight * np.square(confidence)
        return np.column_stack((X, penalty))

def simple_augment_text(text, p=0.1):
    words = text.split()
    augmented_words = [word for word in words if random.random() > p]
    # Ensure we don't delete all words
    if not augmented_words:
        augmented_words = [random.choice(words)]
    return ' '.join(augmented_words)

def compute_augmentation_consistency(model, tokenizer, original_text, augmented_text, max_length):
    # Tokenize both texts with padding and truncation
    original_inputs = tokenizer(original_text,
                                return_tensors='pt',
                                truncation=True,
                                max_length=max_length,
                                padding='max_length').to(model.device)
    augmented_inputs = tokenizer(augmented_text,
                                return_tensors='pt',
                                truncation=True,
                                max_length=max_length,
                                padding='max_length').to(model.device)

    with torch.no_grad():
        original_outputs = model(**original_inputs)
        augmented_outputs = model(**augmented_inputs)

    # Use only the non-padded parts for consistency calculation
    original_probs = torch.softmax(original_outputs.logits, dim=-1)[:, :original_inputs['attention_mask'].sum()]
    augmented_probs = torch.softmax(augmented_outputs.logits, dim=-1)[:, :augmented_inputs['attention_mask'].sum()]

    # Calculate consistency only for the shorter sequence length
    min_length = min(original_probs.size(1), augmented_probs.size(1))
    consistency = torch.mean(torch.abs(original_probs[:, :min_length] - augmented_probs[:, :min_length])).item()

    return consistency

def compute_confidence_error_correlation(logits, true_labels):
    probs = torch.softmax(logits, dim=-1)
    confidence = torch.max(probs, dim=-1)[0]
    predicted_labels = torch.argmax(logits, dim=-1)
    correct = (predicted_labels == true_labels).float()

    # Flatten tensors to 1D
    confidence = confidence.view(-1)
    correct = correct.view(-1)

    # Compute correlation
    if confidence.numel() > 1:  # Check if we have more than one element
        correlation = torch.corrcoef(torch.stack([confidence, correct]))[0, 1].item()
    else:
        # If we have only one element, correlation is not defined
        correlation = 0.0

    return correlation

def compute_memorization_score(loss, batch_loss):
    return (loss - batch_loss.mean()).item()

def compute_logit_margin(logits):
    sorted_logits, _ = torch.sort(logits, dim=-1, descending=True)
    margin = (sorted_logits[:, 0] - sorted_logits[:, 1]).mean().item()
    return margin

def compute_contrastive_loss(embeddings, other_embeddings, temperature=0.5):
    embeddings = embeddings.mean(dim=1)  # Average over sequence length
    other_embeddings = other_embeddings.mean(dim=1)

    similarity = torch.mm(embeddings, other_embeddings.t()) / temperature
    labels = torch.arange(similarity.size(0)).to(embeddings.device)

    loss = torch.nn.functional.cross_entropy(similarity, labels)
    return loss.item()

def compute_ood_score(logits, probs):
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()
    max_prob = torch.max(probs, dim=-1)[0].mean().item()
    return entropy, max_prob

class LanguageMIA:
    def __init__(self):
        # Initialize any necessary attributes here
        pass

    def attack(self, train_dataset, test_dataset, model, tokenizer):
        # Main attack method
        results = self._improved_membership_inference_attack(train_dataset, test_dataset, model, tokenizer)
        return results

    def get_mia_features(self, dataset, model, tokenizer, max_length=128):
        features = []
        model.gradient_checkpointing_enable()

        total_samples = len(dataset)
        print(f"Total number of samples to process: {total_samples}")

        for i, batch in enumerate(dataset):
            with torch.enable_grad():
                text = batch['text'] if isinstance(batch, dict) and 'text' in batch else batch
                inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(model.device)

                embedding_layer = model.get_input_embeddings()
                embeddings = embedding_layer(inputs['input_ids']).detach()
                embeddings = embeddings.clone().requires_grad_(True)

                outputs = model(inputs_embeds=embeddings, attention_mask=inputs['attention_mask'])
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                probs = torch.softmax(logits, dim=-1)

                # Existing features
                max_prob = torch.max(probs, dim=-1)[0].mean().item()
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()
                perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10), dim=-1)).mean().item()
                std_prob = torch.std(probs, dim=-1).mean().item()
                kl_uniform = torch.sum(probs * torch.log(probs * probs.size(-1) + 1e-10), dim=-1).mean().item()

                # New features
                # 1. Data Augmentation Responses
                augmented_text = simple_augment_text(text)
                aug_consistency = compute_augmentation_consistency(model, tokenizer, text, augmented_text, max_length)

                # 2. Memorization Metrics
                conf_error_corr = compute_confidence_error_correlation(logits, inputs['input_ids'])
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), inputs['input_ids'].view(-1))
                mem_score = compute_memorization_score(loss, loss.detach())

                # 3. Generalization Indicators
                logit_margin = compute_logit_margin(logits)

                # 4. Contrastive Learning Features
                other_inputs = tokenizer(random.choice(dataset)['text'], return_tensors='pt', truncation=True, max_length=max_length).to(model.device)
                other_embeddings = embedding_layer(other_inputs['input_ids']).detach()
                contrastive_loss = compute_contrastive_loss(embeddings, other_embeddings)

                # 5. Out of Distribution Detection Scores
                ood_entropy, ood_max_prob = compute_ood_score(logits, probs)

                # Existing additional features
                top_k = 5
                top_k_probs, _ = torch.topk(probs, k=top_k, dim=-1)
                top_k_entropy = -torch.sum(top_k_probs * torch.log(top_k_probs + 1e-10), dim=-1).mean().item()
                top_k_ppl = torch.exp(-torch.sum(top_k_probs * torch.log(top_k_probs + 1e-10), dim=-1)).mean().item()

                prob_sample = probs.view(-1)
                if prob_sample.numel() > 10000:
                    prob_sample = prob_sample[:10000]
                quantiles = torch.tensor([0.25, 0.5, 0.75], device=probs.device)
                prob_percentiles = torch.quantile(prob_sample, quantiles).mean().item()

                att_entropy = 0
                if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    att_weights = torch.stack(outputs.attentions).mean(dim=[0,1])
                    att_entropy = -torch.sum(att_weights * torch.log(att_weights + 1e-10), dim=-1).mean().item()

                sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
                uncertainty = (sorted_probs[:, 0] - sorted_probs[:, 1]).mean().item()

                features.append([
                    max_prob, entropy, perplexity, std_prob, kl_uniform,
                    top_k_entropy, top_k_ppl, att_entropy, uncertainty, prob_percentiles,
                    aug_consistency, conf_error_corr, mem_score, logit_margin,
                    contrastive_loss, ood_entropy, ood_max_prob
                ])

                model.zero_grad()
                loss.backward()

                # Per-layer gradient norms
                layer_grad_norms = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        layer_grad_norms.append(torch.norm(param.grad).item())
                features[-1].extend(layer_grad_norms)

                del embeddings, outputs, logits, probs
                torch.cuda.empty_cache()

            if (i + 1) % 50 == 0 or (i + 1) == total_samples:
                print(f"Processed {i + 1}/{total_samples} samples")

        print(f"Total number of features extracted: {len(features[0])}")

        return np.array(features)

    def _improved_membership_inference_attack(self, train_dataset, test_dataset, model, tokenizer):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Feature extraction
        train_features = self.get_mia_features(train_dataset, model, tokenizer, max_length=512)
        test_features = self.get_mia_features(test_dataset, model, tokenizer, max_length=512)

        print(f"Shape of train_features: {train_features.shape}")
        print(f"Shape of test_features: {test_features.shape}")

        # Create labels for all processed samples
        train_labels = np.ones(len(train_dataset))
        test_labels = np.zeros(len(test_dataset))

        print(f"Shape of train_labels: {train_labels.shape}")
        print(f"Shape of test_labels: {test_labels.shape}")

        # Check for any discrepancy between the number of features and labels
        assert train_features.shape[0] == train_labels.shape[0], "Mismatch between train features and labels"
        assert test_features.shape[0] == test_labels.shape[0], "Mismatch between test features and labels"

        # Combine features and labels
        X = np.vstack((train_features, test_features))
        y = np.concatenate((train_labels, test_labels))

        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape}")

        unique, counts = np.unique(y, return_counts=True)
        print("Class distribution:", dict(zip(unique, counts)))

        # Split the data before preprocessing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Preprocessing on training data
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)

        # Feature selection using mutual information on training data only
        mi_scores = mutual_info_classif(X_train_scaled, y_train)
        top_features = np.argsort(mi_scores)[-20:]  # Select top 20 features
        X_train_selected = X_train_scaled[:, top_features]
        X_test_selected = X_test_scaled[:, top_features]

        # Visualize selected features
        visualize_features(X_train_selected, y_train, "Selected Features (Train)")

        # Create polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_selected)
        X_test_poly = poly.transform(X_test_selected)

        # Apply SMOTE for class balancing on training data only
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_poly, y_train)

        classifiers = [
            (RandomForestClassifier(random_state=42, class_weight='balanced'), "Random Forest", {
                'n_estimators': randint(100, 500),
                'max_depth': randint(5, 30),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 20)
            }),
            (LogisticRegression(random_state=42, class_weight='balanced'), "Logistic Regression", {
                'C': uniform(0.1, 10),
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }),
            (MLPClassifier(random_state=42), "Neural Network", {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['tanh', 'relu'],
                'alpha': uniform(0.0001, 0.01),
                'learning_rate': ['constant', 'adaptive']
            })
        ]

        best_models = []

        # Use StratifiedKFold for cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for clf, name, param_dist in classifiers:
            print(f"\nTuning {name}...")
            pipeline = Pipeline([
                ('confidence_penalty', ConfidencePenaltyTransformer()),
                ('classifier', clf)
            ])

            with parallel_backend('loky'):
                random_search = RandomizedSearchCV(
                    pipeline,
                    param_distributions={'classifier__' + key: value for key, value in param_dist.items()},
                    n_iter=50,
                    cv=skf,
                    scoring='roc_auc',
                    n_jobs=-1,
                    random_state=42
                )
                random_search.fit(X_train_resampled, y_train_resampled)

            best_clf = random_search.best_estimator_
            y_pred = best_clf.predict(X_test_poly)
            y_pred_proba = best_clf.predict_proba(X_test_poly)[:, 1]

            roc_auc = roc_auc_score(y_test, y_pred_proba)
            brier_score = brier_score_loss(y_test, y_pred_proba)

            print(f"{name} results:")
            print(f"Best parameters: {random_search.best_params_}")
            print(f"ROC AUC: {roc_auc:.4f}")
            print(f"Brier Score: {brier_score:.4f}")

            best_models.append((best_clf, name, roc_auc))

            # Plot learning curve
            plot_learning_curve(best_clf, X_train_resampled, y_train_resampled, f"Learning Curve - {name}")

        # Stacking ensemble with cross-validation
        base_models = [model for model, _, _ in best_models]
        stacking_clf = StackingClassifier(estimators=[(f"model_{i}", model) for i, model in enumerate(base_models)],
                                        final_estimator=LogisticRegression(class_weight='balanced'),
                                        cv=skf)
        stacking_clf.fit(X_train_resampled, y_train_resampled)
        y_pred_stacking = stacking_clf.predict(X_test_poly)
        y_pred_proba_stacking = stacking_clf.predict_proba(X_test_poly)[:, 1]
        roc_auc_stacking = roc_auc_score(y_test, y_pred_proba_stacking)
        brier_score_stacking = brier_score_loss(y_test, y_pred_proba_stacking)

        print("\nStacking Ensemble results:")
        print(f"ROC AUC: {roc_auc_stacking:.4f}")
        print(f"Brier Score: {brier_score_stacking:.4f}")

        # Select the best model (including stacking)
        best_models.append((stacking_clf, "Stacking Ensemble", roc_auc_stacking))
        best_model, best_name, best_score = max(best_models, key=lambda x: x[2])

        print(f"\nBest model: {best_name} with ROC AUC: {best_score:.4f}")

        y_pred = best_model.predict(X_test_poly)
        y_pred_proba = best_model.predict_proba(X_test_poly)[:, 1]

        # Calibrate probabilities
        calibrated_model = CalibratedClassifierCV(best_model, cv='prefit')
        calibrated_model.fit(X_test_poly, y_test)
        y_pred_proba_calibrated = calibrated_model.predict_proba(X_test_poly)[:, 1]

        # Calculate metrics with standard threshold (0.5)
        y_pred_standard = (y_pred_proba_calibrated >= 0.5).astype(int)

        # Find optimal threshold
        optimal_threshold = find_optimal_threshold(y_test, y_pred_proba_calibrated)
        y_pred_optimal = (y_pred_proba_calibrated >= optimal_threshold).astype(int)

        # Perform error analysis for both standard and optimal thresholds
        print("\nError Analysis (Standard Threshold):")
        perform_error_analysis(X_test_poly, y_test, y_pred_standard, feature_names=np.arange(X_test_poly.shape[1]))
        print("\nError Analysis (Optimal Threshold):")
        perform_error_analysis(X_test_poly, y_test, y_pred_optimal, feature_names=np.arange(X_test_poly.shape[1]))

        # Get feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = best_model.feature_importances_
        elif hasattr(best_model, 'coef_'):
            feature_importance = np.abs(best_model.coef_[0])
        else:
            feature_importance = None

        if feature_importance is not None:
            plot_feature_importance(feature_importance, np.arange(X_test_poly.shape[1]), f"Feature Importance - {best_name}")

        # Calculate additional metrics
        precision_recall_auc = average_precision_score(y_test, y_pred_proba_calibrated)
        brier_score = brier_score_loss(y_test, y_pred_proba_calibrated)

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_calibrated)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')
        plt.close()

        return {
            "roc_auc": roc_auc,
            "precision_recall_auc": precision_recall_auc,
            "brier_score": brier_score,
            "best_model": best_name,
            "optimal_threshold": optimal_threshold,
            "feature_importance": feature_importance,
            # Metrics with standard threshold (0.5)
            "accuracy_standard": accuracy_score(y_test, y_pred_standard),
            "precision_standard": precision_score(y_test, y_pred_standard),
            "recall_standard": recall_score(y_test, y_pred_standard),
            "f1_standard": f1_score(y_test, y_pred_standard),
            # Metrics with optimal threshold
            "accuracy_optimal": accuracy_score(y_test, y_pred_optimal),
            "precision_optimal": precision_score(y_test, y_pred_optimal),
            "recall_optimal": recall_score(y_test, y_pred_optimal),
            "f1_optimal": f1_score(y_test, y_pred_optimal),
            # Additional information
            "n_samples": len(y_test),
            "n_positive": sum(y_test),
            "n_negative": len(y_test) - sum(y_test),
        }

class ImageMIA(LanguageMIA):
    def __init__(self):
        super().__init__()

    def attack(self, train_dataset, test_dataset, model, _):
        # Override to handle image datasets
        return self._improved_membership_inference_attack(train_dataset, test_dataset, model, None)

    def _improved_membership_inference_attack(self, train_dataset, test_dataset, model, _):
        # Override to handle image-specific preprocessing if needed
        return super()._improved_membership_inference_attack(train_dataset, test_dataset, model, None)

    @staticmethod
    def simple_augment_image(image, p=0.1):
        # Simple image augmentation (e.g., random noise)
        return image + torch.randn_like(image) * p

    # Override or adapt other helper methods as needed
    def compute_confidence_error_correlation(self, logits, true_labels):
        # Adapt for image data if necessary
        return super().compute_confidence_error_correlation(logits, true_labels)

    def compute_memorization_score(self, loss, batch_loss):
        # This might not need adaptation for images
        return super().compute_memorization_score(loss, batch_loss)

    def compute_logit_margin(self, logits):
        # This might not need adaptation for images
        return super().compute_logit_margin(logits)

    def compute_contrastive_loss(self, embeddings, other_embeddings, temperature=0.5):
        # Adapt for image embeddings if necessary
        return super().compute_contrastive_loss(embeddings, other_embeddings, temperature)

    def compute_ood_score(self, logits, probs):
        # This might not need adaptation for images
        return super().compute_ood_score(logits, probs)

    def get_mia_features(self, dataset, model, _, max_length=None):
        features = []
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        model.eval()
        
        # Get the device from the model's parameters
        device = next(model.parameters()).device

        total_samples = len(dataset)
        print(f"Total number of samples to process: {total_samples}")

        for i, (data, _) in enumerate(dataloader):
            with torch.enable_grad():
                data = data.to(device)
                
                # Get embeddings (assuming the model has a feature extractor)
                if hasattr(model, 'get_input_embeddings'):
                    embedding_layer = model.get_input_embeddings()
                    embeddings = embedding_layer(data).detach()
                else:
                    # If no embedding layer, use the first convolutional layer as feature extractor
                    first_conv = next(model.modules())
                    if isinstance(first_conv, torch.nn.Conv2d):
                        embeddings = first_conv(data)
                    else:
                        embeddings = data  # Fallback to using raw input if no suitable layer found
                
                embeddings = embeddings.clone().requires_grad_(True)

                outputs = model(embeddings)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                probs = torch.softmax(logits, dim=1)

                batch_features = []
                for prob in probs:
                    # Existing features
                    max_prob = torch.max(prob).item()
                    entropy = (-prob * torch.log(prob + 1e-10)).sum().item()
                    perplexity = torch.exp(entropy)
                    std_prob = torch.std(prob).item()
                    kl_uniform = (prob * torch.log(prob * prob.size(0) + 1e-10)).sum().item()

                    # Additional features
                    top_k = 5
                    top_k_probs, _ = torch.topk(prob, k=top_k)
                    top_k_entropy = (-top_k_probs * torch.log(top_k_probs + 1e-10)).sum().item()
                    top_k_ppl = torch.exp((-top_k_probs * torch.log(top_k_probs + 1e-10)).sum()).item()

                    sorted_probs, _ = torch.sort(prob, descending=True)
                    uncertainty = (sorted_probs[0] - sorted_probs[1]).item()

                    # Adapt other features
                    # 1. Data Augmentation Responses (simplified for images)
                    augmented_data = self.simple_augment_image(data)
                    aug_outputs = model(augmented_data)
                    aug_probs = torch.softmax(aug_outputs.logits if hasattr(aug_outputs, 'logits') else aug_outputs, dim=1)
                    aug_consistency = torch.mean(torch.abs(probs - aug_probs)).item()

                    # 2. Memorization Metrics
                    conf_error_corr = self.compute_confidence_error_correlation(logits, torch.argmax(probs, dim=1))
                    loss = torch.nn.functional.cross_entropy(logits, torch.argmax(probs, dim=1))
                    mem_score = self.compute_memorization_score(loss, loss.detach())

                    # 3. Generalization Indicators
                    logit_margin = self.compute_logit_margin(logits)

                    # 4. Contrastive Learning Features
                    other_data = random.choice(dataset)[0].unsqueeze(0).to(model.device)
                    other_embeddings = model.get_input_embeddings()(other_data) if hasattr(model, 'get_input_embeddings') else other_data
                    contrastive_loss = self.compute_contrastive_loss(embeddings, other_embeddings)

                    # 5. Out of Distribution Detection Scores
                    ood_entropy, ood_max_prob = self.compute_ood_score(logits, probs)

                    sample_features = [
                        max_prob, entropy, perplexity, std_prob, kl_uniform,
                        top_k_entropy, top_k_ppl, uncertainty, aug_consistency,
                        conf_error_corr, mem_score, logit_margin,
                        contrastive_loss, ood_entropy, ood_max_prob
                    ]

                    batch_features.append(sample_features)

                # Compute gradient norms
                loss = torch.mean(torch.sum(-probs * torch.log(probs + 1e-10), dim=1))
                loss.backward()

                layer_grad_norms = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        layer_grad_norms.append(torch.norm(param.grad).item())

                for sample_features in batch_features:
                    sample_features.extend(layer_grad_norms)

                features.extend(batch_features)

                model.zero_grad()

            if (i + 1) % 10 == 0 or (i + 1) * dataloader.batch_size >= total_samples:
                print(f"Processed {min((i + 1) * dataloader.batch_size, total_samples)}/{total_samples} samples")

        print(f"Total number of features extracted: {len(features[0])}")

        return np.array(features)