import net
import torch
from face_alignment import align
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
from inference import load_pretrained_model, to_input

adaface_models = {
    'ir_50':"pretrained/adaface_ir50_casia.ckpt",
}

def get_lfw_image_path(lfw_dir, name, imagenum):
    """Get LFW image path"""
    image_name = f"{name}_{imagenum:04d}.jpg"
    return os.path.join(lfw_dir, name, image_name)


def extract_feature(model, image_path, use_flip=False):
    """Extract feature with optional flip augmentation"""
    if not os.path.exists(image_path):
        return None
    
    try:
        # Align face
        aligned_rgb_img = align.get_aligned_face(image_path)
        if aligned_rgb_img is None:
            return None
        
        # Extract feature from aligned image
        bgr_tensor_input = to_input(aligned_rgb_img)
        
        with torch.no_grad():
            feature, _ = model(bgr_tensor_input)
        
        feature = feature.squeeze().numpy()
        
        # Flip augmentation
        if use_flip:
            aligned_flipped = aligned_rgb_img.transpose(1)  # Flip horizontally (PIL Image)
            bgr_tensor_flipped = to_input(aligned_flipped)
            
            with torch.no_grad():
                feature_flipped, _ = model(bgr_tensor_flipped)
            
            feature_flipped = feature_flipped.squeeze().numpy()
            
            # Average features
            feature = (feature + feature_flipped) / 2.0
        
        # Normalize L2
        feature = feature / np.linalg.norm(feature)
        return feature
        
    except Exception as e:
        return None


def compute_similarity(f1, f2):
    """Compute cosine similarity"""
    if f1 is None or f2 is None:
        return None
    
    # Normalize again to be safe
    f1 = f1 / np.linalg.norm(f1)
    f2 = f2 / np.linalg.norm(f2)
    
    similarity = np.dot(f1, f2)
    return similarity


def parse_lfw_pairs(pairs_file):
    """Parse LFW pairs.csv (supports 3-column positive rows and 4-column negative rows).

    Formats (comma-separated, trailing commas allowed):
    - Same person: name, imagenum1, imagenum2[,]
    - Different:   name1, imagenum1, name2, imagenum2
    Header row is ignored if present.
    """
    pairs = []

    with open(pairs_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # Skip header if present (first token is usually "name")
            if line_idx == 0 and line.lower().startswith('name'):
                continue

            # Split by comma, drop empty fields (handles trailing commas)
            parts = [p.strip() for p in line.split(',') if p.strip()]
            if not parts:
                continue

            if len(parts) == 3:
                # Same person: name, imagenum1, imagenum2
                name = parts[0]
                try:
                    imagenum1 = int(parts[1])
                    imagenum2 = int(parts[2])
                except ValueError:
                    continue  # Skip malformed rows
                pairs.append((1, name, imagenum1, name, imagenum2))
            elif len(parts) == 4:
                # Different person: name1, imagenum1, name2, imagenum2
                name1 = parts[0]
                name2 = parts[2]
                try:
                    imagenum1 = int(parts[1])
                    imagenum2 = int(parts[3])
                except ValueError:
                    continue  # Skip malformed rows
                pairs.append((0, name1, imagenum1, name2, imagenum2))
            else:
                # Unexpected column count; skip
                continue

    return pairs


def split_10folds(pairs, num_folds=10):
    """Split pairs into 10 folds
    Ensure balanced distribution of positive/negative pairs in each fold
    """
    # Separate positive and negative pairs
    positive_pairs = [p for p in pairs if p[0] == 1]
    negative_pairs = [p for p in pairs if p[0] == 0]
    
    print(f"Total positive pairs: {len(positive_pairs)}")
    print(f"Total negative pairs: {len(negative_pairs)}")
    
    # Split each group into 10 folds
    folds_pos = [[] for _ in range(num_folds)]
    folds_neg = [[] for _ in range(num_folds)]
    
    # Distribute positive pairs
    for i, pair in enumerate(positive_pairs):
        fold_idx = i % num_folds
        folds_pos[fold_idx].append(pair)
    
    # Distribute negative pairs
    for i, pair in enumerate(negative_pairs):
        fold_idx = i % num_folds
        folds_neg[fold_idx].append(pair)
    
    # Combine into 10 folds
    folds = []
    for i in range(num_folds):
        fold = folds_pos[i] + folds_neg[i]
        folds.append(fold)
        print(f"Fold {i+1}: {len(folds_pos[i])} pos + {len(folds_neg[i])} neg = {len(fold)} total")
    
    return folds


def evaluate_fold(model, lfw_dir, fold_pairs, use_flip=False):
    """Evaluate on a single fold"""
    similarities = []
    labels = []
    failed_pairs = 0
    
    for pair in tqdm(fold_pairs, desc="Processing pairs", leave=False):
        label, name1, imagenum1, name2, imagenum2 = pair
        
        path1 = get_lfw_image_path(lfw_dir, name1, imagenum1)
        path2 = get_lfw_image_path(lfw_dir, name2, imagenum2)
        
        feature1 = extract_feature(model, path1, use_flip=use_flip)
        feature2 = extract_feature(model, path2, use_flip=use_flip)
        
        similarity = compute_similarity(feature1, feature2)
        
        if similarity is not None:
            similarities.append(similarity)
            labels.append(label)
        else:
            failed_pairs += 1
    
    if len(similarities) == 0:
        return None, 0, 0
    
    # Find best threshold on this fold
    similarities = np.array(similarities)
    labels = np.array(labels)
    
    best_acc = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.0, 1.0, 0.01):
        predictions = (similarities > threshold).astype(int)
        accuracy = np.mean(predictions == labels)
        
        if accuracy > best_acc:
            best_acc = accuracy
            best_threshold = threshold
    
    return best_acc, best_threshold, failed_pairs


def main():
    print("="*80)
    print("AdaFace LFW 6000-pair 10-fold evaluation")
    print("="*80)
    
    # Configuration
    model_path = 'pretrained/adaface_ir50_casia.ckpt'
    lfw_dir = 'datasets/lfw-deepfunneled/lfw-deepfunneled'
    pairs_file = 'datasets/pairs.csv'
    
    use_flip = True  # Use flip augmentation
    
    # Load model
    print(f"\n→ Loading model from {model_path}...")
    model = load_pretrained_model('ir_50')
    print("✓ Model loaded!")
    
    # Parse pairs
    print(f"\n→ Parsing pairs from {pairs_file}...")
    pairs = parse_lfw_pairs(pairs_file)
    print(f"✓ Loaded {len(pairs)} pairs")
    
    # Split into 10 folds
    print(f"\n→ Splitting into 10 folds...")
    folds = split_10folds(pairs, num_folds=10)
    
    # Evaluate on each fold
    print(f"\n→ Evaluating on 10 folds{'(with flip augmentation)' if use_flip else ''}...")
    fold_accuracies = []
    fold_thresholds = []
    
    for fold_idx, fold_pairs in enumerate(folds):
        print(f"\nFold {fold_idx + 1}/10:")
        accuracy, threshold, failed = evaluate_fold(model, lfw_dir, fold_pairs, use_flip=use_flip)
        
        if accuracy is not None:
            fold_accuracies.append(accuracy)
            fold_thresholds.append(threshold)
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Best Threshold: {threshold:.4f}")
            print(f"  Failed pairs: {failed}")
        else:
            print(f"  Error: No valid pairs in fold")
    
    # Calculate mean accuracy
    if len(fold_accuracies) > 0:
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        mean_threshold = np.mean(fold_thresholds)
        
        print("\n" + "="*80)
        print("FINAL RESULTS (10-FOLD CROSS-VALIDATION)")
        print("="*80)
        print(f"\nMean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"Mean Threshold: {mean_threshold:.4f}")
        print(f"\nPer-fold accuracies:")
        for i, acc in enumerate(fold_accuracies):
            print(f"  Fold {i+1:2d}: {acc:.4f}")
        print("="*80)
        
        # Save results
        with open('lfw_10fold_results.txt', 'w') as f:
            f.write("AdaFace LFW 6000-pair 10-fold Evaluation\n")
            f.write("="*80 + "\n\n")
            f.write(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}\n")
            f.write(f"Mean Threshold: {mean_threshold:.4f}\n")
            f.write(f"Use flip augmentation: {use_flip}\n\n")
            f.write("Per-fold accuracies:\n")
            for i, acc in enumerate(fold_accuracies):
                f.write(f"  Fold {i+1:2d}: {acc:.4f}\n")
        
        print("\n✓ Results saved to lfw_10fold_results.txt")
    else:
        print("✗ No valid folds")


if __name__ == '__main__':
    main()
