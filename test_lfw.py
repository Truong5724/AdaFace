import net
import torch
from face_alignment import align
import numpy as np
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from inference import load_pretrained_model, to_input

adaface_models = {
    'ir_50':"pretrained/adaface_ir50_casia.ckpt",
}

def get_lfw_image_path(lfw_dir, name, imagenum):
    """Get full path to LFW image"""
    # Format: PersonName/PersonName_0001.jpg
    image_name = f"{name}_{imagenum:04d}.jpg"
    image_path = os.path.join(lfw_dir, name, image_name)
    return image_path


def extract_feature(model, image_path):
    """Extract feature from an image"""
    if not os.path.exists(image_path):
        return None
    
    try:
        # Align face
        aligned_rgb_img = align.get_aligned_face(image_path)
        if aligned_rgb_img is None:
            return None
        
        # Convert to tensor
        bgr_tensor_input = to_input(aligned_rgb_img)
        
        # Extract features
        with torch.no_grad():
            feature, _ = model(bgr_tensor_input)
        
        return feature.squeeze().numpy()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def compute_similarity(feature1, feature2):
    """Compute cosine similarity between two features"""
    if feature1 is None or feature2 is None:
        return None
    
    # Normalize features
    feature1 = feature1 / np.linalg.norm(feature1)
    feature2 = feature2 / np.linalg.norm(feature2)
    
    # Cosine similarity
    similarity = np.dot(feature1, feature2)
    return similarity


def evaluate_lfw_pairs(model, lfw_dir, pairs_file, is_match=True):
    """Evaluate on LFW pairs
    
    Args:
        model: AdaFace model
        lfw_dir: Directory containing LFW images
        pairs_file: CSV file with pairs
        is_match: True for same person pairs, False for different person pairs
    
    Returns:
        similarities: list of similarity scores
        labels: list of labels (1 for same, 0 for different)
    """
    print(f"\n→ Đang đánh giá: {os.path.basename(pairs_file)}")
    
    # Read pairs
    df = pd.read_csv(pairs_file)
    
    similarities = []
    labels = []
    failed = 0
    
    if is_match:
        # Same person pairs
        label = 1
        total = len(df)
        
        for idx, row in tqdm(df.iterrows(), total=total, desc="Processing"):
            name = row['name']
            imagenum1 = int(row['imagenum1'])
            imagenum2 = int(row['imagenum2'])
            
            path1 = get_lfw_image_path(lfw_dir, name, imagenum1)
            path2 = get_lfw_image_path(lfw_dir, name, imagenum2)
            
            feature1 = extract_feature(model, path1)
            feature2 = extract_feature(model, path2)
            
            similarity = compute_similarity(feature1, feature2)
            
            if similarity is not None:
                similarities.append(similarity)
                labels.append(label)
            else:
                failed += 1
    else:
        # Different person pairs
        label = 0
        total = len(df)
        
        for idx, row in tqdm(df.iterrows(), total=total, desc="Processing"):
            name1 = row.iloc[0]  # First name column
            imagenum1 = int(row.iloc[1])
            name2 = row.iloc[2]  # Second name column
            imagenum2 = int(row.iloc[3])
            
            path1 = get_lfw_image_path(lfw_dir, name1, imagenum1)
            path2 = get_lfw_image_path(lfw_dir, name2, imagenum2)
            
            feature1 = extract_feature(model, path1)
            feature2 = extract_feature(model, path2)
            
            similarity = compute_similarity(feature1, feature2)
            
            if similarity is not None:
                similarities.append(similarity)
                labels.append(label)
            else:
                failed += 1
    
    print(f"✓ Hoàn thành: {len(similarities)} cặp, {failed} cặp thất bại")
    
    return similarities, labels


def compute_metrics(similarities, labels, thresholds=None):
    """Compute evaluation metrics"""
    similarities = np.array(similarities)
    labels = np.array(labels)
    
    if thresholds is None:
        # Use multiple thresholds
        thresholds = np.arange(0.0, 1.0, 0.01)
    
    best_acc = 0
    best_threshold = 0
    
    for threshold in thresholds:
        predictions = (similarities > threshold).astype(int)
        accuracy = np.mean(predictions == labels)
        
        if accuracy > best_acc:
            best_acc = accuracy
            best_threshold = threshold
    
    # Compute accuracy at best threshold
    predictions = (similarities > best_threshold).astype(int)
    
    # True positives, false positives, etc.
    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    # Compute metrics
    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr)
    
    return {
        'accuracy': accuracy,
        'best_threshold': best_threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def plot_roc_curve(fpr, tpr, roc_auc, output_path='lfw_roc_curve.png'):
    """Plot ROC curve"""
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - AdaFace on LFW')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Đã lưu ROC curve: {output_path}")
    plt.close()


def plot_distribution(match_sims, mismatch_sims, output_path='lfw_similarity_distribution.png'):
    """Plot similarity score distribution"""
    plt.figure(figsize=(12, 6))
    
    plt.hist(match_sims, bins=50, alpha=0.5, label='Same Person (Positive)', color='green')
    plt.hist(mismatch_sims, bins=50, alpha=0.5, label='Different Person (Negative)', color='red')
    
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Similarity Score Distribution - AdaFace on LFW')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Đã lưu distribution plot: {output_path}")
    plt.close()


if __name__ == '__main__':
    print("="*80)
    print("ADAFACE EVALUATION ON LFW DATASET")
    print("="*80)
    
    # Configuration
    model_path = 'models/adaface_ir50_casia.ckpt'
    lfw_dir = '../datasets/lfw-deepfunneled/lfw-deepfunneled'
    match_pairs_file = '../datasets/matchpairsDevTest.csv'
    mismatch_pairs_file = '../datasets/mismatchpairsDevTest.csv'
    
    # Load model
    print("\n→ Loading AdaFace model...")
    model = load_pretrained_model('ir_50')
    print("✓ Model loaded successfully!")
    
    # Evaluate on match pairs (same person)
    match_sims, match_labels = evaluate_lfw_pairs(
        model, lfw_dir, match_pairs_file, is_match=True
    )
    
    # Evaluate on mismatch pairs (different person)
    mismatch_sims, mismatch_labels = evaluate_lfw_pairs(
        model, lfw_dir, mismatch_pairs_file, is_match=False
    )
    
    # Combine results
    all_similarities = match_sims + mismatch_sims
    all_labels = match_labels + mismatch_labels
    
    print("\n" + "="*80)
    print("COMPUTING METRICS")
    print("="*80)
    
    # Compute metrics
    metrics = compute_metrics(all_similarities, all_labels)
    
    # Print results
    print("\n" + "="*80)
    print("📊 KẾT QUẢ ĐÁNH GIÁ TRÊN LFW")
    print("="*80)
    print(f"\n{'Metric':<20} {'Value':>15}")
    print("-" * 40)
    print(f"{'Total Pairs':<20} {len(all_similarities):>15,}")
    print(f"{'Match Pairs':<20} {len(match_sims):>15,}")
    print(f"{'Mismatch Pairs':<20} {len(mismatch_sims):>15,}")
    print("-" * 40)
    print(f"{'Accuracy':<20} {metrics['accuracy']:>14.2%}")
    print(f"{'Best Threshold':<20} {metrics['best_threshold']:>15.4f}")
    print(f"{'Precision':<20} {metrics['precision']:>14.2%}")
    print(f"{'Recall':<20} {metrics['recall']:>14.2%}")
    print(f"{'F1-Score':<20} {metrics['f1_score']:>14.2%}")
    print(f"{'ROC AUC':<20} {metrics['roc_auc']:>15.4f}")
    print("-" * 40)
    print(f"{'True Positives':<20} {metrics['tp']:>15,}")
    print(f"{'True Negatives':<20} {metrics['tn']:>15,}")
    print(f"{'False Positives':<20} {metrics['fp']:>15,}")
    print(f"{'False Negatives':<20} {metrics['fn']:>15,}")
    print("="*80)
    
    # Print similarity statistics
    print("\n" + "="*80)
    print("📈 THỐNG KÊ ĐỘ TƯƠNG ĐỒNG")
    print("="*80)
    print(f"\nSame Person Pairs:")
    print(f"  Mean:   {np.mean(match_sims):.4f}")
    print(f"  Median: {np.median(match_sims):.4f}")
    print(f"  Std:    {np.std(match_sims):.4f}")
    print(f"  Min:    {np.min(match_sims):.4f}")
    print(f"  Max:    {np.max(match_sims):.4f}")
    
    print(f"\nDifferent Person Pairs:")
    print(f"  Mean:   {np.mean(mismatch_sims):.4f}")
    print(f"  Median: {np.median(mismatch_sims):.4f}")
    print(f"  Std:    {np.std(mismatch_sims):.4f}")
    print(f"  Min:    {np.min(mismatch_sims):.4f}")
    print(f"  Max:    {np.max(mismatch_sims):.4f}")
    print("="*80)
    
    # Plot ROC curve
    print("\n→ Plotting ROC curve...")
    plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc'])
    
    # Plot similarity distribution
    print("→ Plotting similarity distribution...")
    plot_distribution(match_sims, mismatch_sims)
    
    # Save results to file
    print("\n→ Saving results to file...")
    with open('lfw_results.txt', 'w', encoding='utf-8') as f:
        f.write("ADAFACE EVALUATION ON LFW DATASET\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total Pairs: {len(all_similarities)}\n")
        f.write(f"Match Pairs: {len(match_sims)}\n")
        f.write(f"Mismatch Pairs: {len(mismatch_sims)}\n\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Best Threshold: {metrics['best_threshold']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
        f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n\n")
        f.write(f"True Positives: {metrics['tp']}\n")
        f.write(f"True Negatives: {metrics['tn']}\n")
        f.write(f"False Positives: {metrics['fp']}\n")
        f.write(f"False Negatives: {metrics['fn']}\n")
    
    print("✓ Đã lưu kết quả: lfw_results.txt")
    
    print("\n" + "="*80)
    print("✓ HOÀN THÀNH ĐÁNH GIÁ!")
    print("="*80)
