import net
import torch
from face_alignment import align
import numpy as np
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
from inference import load_pretrained_model, to_input

adaface_models = {
    'ir_50': "pretrained/adaface_ir50_casia.ckpt",
}


def extract_feature(model, image_path, base_dir):
    """Extract feature from an image"""
    # Construct full path
    full_path = os.path.join(base_dir, image_path)
    
    if not os.path.exists(full_path):
        return None
    
    try:
        # Align face
        aligned_rgb_img = align.get_aligned_face(full_path)
        if aligned_rgb_img is None:
            return None
        
        # Convert to tensor
        bgr_tensor_input = to_input(aligned_rgb_img)
        
        # Extract features
        with torch.no_grad():
            feature, _ = model(bgr_tensor_input)
        
        return feature.squeeze().numpy()
    except Exception as e:
        print(f"Error processing {full_path}: {e}")
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


def evaluate_casia_pairs(model, base_dir, pairs_file):
    """Evaluate on CASIA-WebFace-10K pairs
    
    Args:
        model: AdaFace model
        base_dir: Base directory containing CASIA-WebFace-10K
        pairs_file: CSV file with pairs (image1, image2, label)
    
    Returns:
        similarities: list of similarity scores
        labels: list of labels (1 for same person, 0 for different person)
    """
    print(f"\n→ Đang đánh giá: {os.path.basename(pairs_file)}")
    
    # Read pairs
    df = pd.read_csv(pairs_file)
    
    positive_similarities = []
    negative_similarities = []
    all_similarities = []
    all_labels = []
    failed = 0
    total = len(df)
    
    for idx, row in tqdm(df.iterrows(), total=total, desc="Processing"):
        image1_path = row['image1']
        image2_path = row['image2']
        label = int(row['label'])
        
        feature1 = extract_feature(model, image1_path, base_dir)
        feature2 = extract_feature(model, image2_path, base_dir)
        
        similarity = compute_similarity(feature1, feature2)
        
        if similarity is not None:
            all_similarities.append(similarity)
            all_labels.append(label)
            
            if label == 1:
                positive_similarities.append(similarity)
            else:
                negative_similarities.append(similarity)
        else:
            failed += 1
    
    print(f"✓ Hoàn thành: {len(all_similarities)} cặp, {failed} cặp thất bại")
    print(f"  - Positive pairs: {len(positive_similarities)}")
    print(f"  - Negative pairs: {len(negative_similarities)}")
    
    return all_similarities, all_labels, positive_similarities, negative_similarities


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
    
    return {
        'accuracy': accuracy,
        'best_threshold': best_threshold,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


if __name__ == '__main__':
    print("="*80)
    print("ADAFACE EVALUATION ON CASIA-WebFace-10K DATASET")
    print("="*80)
    
    # Configuration
    casia_base_dir = 'CASIA-WebFace-10K'
    pairs_file = os.path.join(casia_base_dir, 'pairs.csv')
    
    # Load model
    print("\n→ Loading AdaFace model...")
    model = load_pretrained_model('ir_50')
    print("✓ Model loaded successfully!")
    
    # Evaluate on pairs
    similarities, labels, pos_sims, neg_sims = evaluate_casia_pairs(
        model, casia_base_dir, pairs_file
    )
    
    print("\n" + "="*80)
    print("COMPUTING METRICS")
    print("="*80)
    
    # Compute metrics
    metrics = compute_metrics(similarities, labels)
    
    # Print results
    print("\n" + "="*80)
    print("📊 KẾT QUẢ ĐÁNH GIÁ TRÊN CASIA-WebFace-10K")
    print("="*80)
    print(f"\n{'Metric':<20} {'Value':>15}")
    print("-" * 40)
    print(f"{'Total Pairs':<20} {len(similarities):>15,}")
    print(f"{'Positive Pairs':<20} {len(pos_sims):>15,}")
    print(f"{'Negative Pairs':<20} {len(neg_sims):>15,}")
    print("-" * 40)
    print(f"{'Accuracy':<20} {metrics['accuracy']:>14.2%}")
    print(f"{'Best Threshold':<20} {metrics['best_threshold']:>15.4f}")
    print("-" * 40)
    print(f"{'True Positives':<20} {metrics['tp']:>15,}")
    print(f"{'True Negatives':<20} {metrics['tn']:>15,}")
    print(f"{'False Positives':<20} {metrics['fp']:>15,}")
    print(f"{'False Negatives':<20} {metrics['fn']:>15,}")
    print("="*80)
    
    # Save results to file
    print("\n→ Saving results to file...")
    with open('casia_results.txt', 'w', encoding='utf-8') as f:
        f.write("ADAFACE EVALUATION ON CASIA-WebFace-10K DATASET\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total Pairs: {len(similarities)}\n")
        f.write(f"Positive Pairs: {len(pos_sims)}\n")
        f.write(f"Negative Pairs: {len(neg_sims)}\n\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Best Threshold: {metrics['best_threshold']:.4f}\n")
        f.write(f"TP: {metrics['tp']}, TN: {metrics['tn']}, FP: {metrics['fp']}, FN: {metrics['fn']}\n")
    
    print("✓ Đã lưu kết quả: casia_results.txt")
    
    print("\n" + "="*80)
    print("✓ HOÀN THÀNH ĐÁNH GIÁ!")
    print("="*80)
