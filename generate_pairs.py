import os
import csv
import random
from collections import defaultdict
import pandas as pd


def generate_pairs_from_labels(labels_csv_path, output_pairs_csv, num_positive_pairs=1000, num_negative_pairs=1000):
    """
    Tạo file pairs.csv từ labels.csv để sử dụng kiểm thử face recognition
    
    Args:
        labels_csv_path: Đường dẫn đến file labels.csv
        output_pairs_csv: Đường dẫn để lưu file pairs.csv
        num_positive_pairs: Số lượng positive pairs (cùng người). Default: 1000
        num_negative_pairs: Số lượng negative pairs (khác người). Default: 1000
    
    Returns:
        Số lượng positive pairs và negative pairs được tạo
    """
    random.seed(57)  # Set seed để kết quả lặp lại được
    
    # Đọc file labels
    print(f"Đang đọc file labels từ {labels_csv_path}...")
    df = pd.read_csv(labels_csv_path)
    
    # Gom các ảnh theo label (ID khuôn mặt)
    label_to_images = defaultdict(list)
    for row in df.itertuples(index=False):
        label_to_images[row.label].append(row.image_path)
    
    print(f"Tổng số người: {len(label_to_images)}")
    print(f"Tổng số ảnh: {len(df)}")
    
    # Tạo positive pairs (sử dụng 100% dữ liệu)
    positive_pairs = []
    
    for label, test_images in label_to_images.items():
        if len(test_images) >= 2:
            # Tạo tất cả các cặp từ images của cùng một người
            for i in range(len(test_images)):
                for j in range(i + 1, len(test_images)):
                    positive_pairs.append((test_images[i], test_images[j], 1))
    
    print(f"Tổng số positive pairs có thể tạo: {len(positive_pairs)}")
    
    # Giới hạn số positive pairs
    if len(positive_pairs) > num_positive_pairs:
        positive_pairs = random.sample(positive_pairs, num_positive_pairs)
    
    num_positive = len(positive_pairs)
    
    # Tạo mapping ảnh -> label và danh sách tất cả ảnh
    label_for_images = {img: label for label, images in label_to_images.items() for img in images}
    test_images_flat = list(label_for_images.keys())
    
    # Tạo negative pairs
    negative_pairs = []
    attempts = 0
    max_attempts = num_negative_pairs * 10
    
    while len(negative_pairs) < num_negative_pairs and attempts < max_attempts:
        img1 = random.choice(test_images_flat)
        img2 = random.choice(test_images_flat)
        
        if img1 != img2 and label_for_images[img1] != label_for_images[img2]:
            negative_pairs.append((img1, img2, 0))
        
        attempts += 1
    
    print(f"Tổng số negative pairs được tạo: {len(negative_pairs)}")
    
    # Ghi vào file CSV
    print(f"Ghi dữ liệu vào {output_pairs_csv}...")
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    
    with open(output_pairs_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image1', 'image2', 'label'])  # Header
        for pair in all_pairs:
            writer.writerow(pair)
    
    print(f"✓ Hoàn thành! Tạo {num_positive} positive pairs và {len(negative_pairs)} negative pairs")
    print(f"✓ File pairs đã lưu tại: {output_pairs_csv}")
    
    return num_positive, len(negative_pairs)


if __name__ == "__main__":
    # Thư mục CASIA-WebFace-10K
    base_dir = "CASIA-WebFace-10K"
    labels_csv = os.path.join(base_dir, "labels.csv")
    output_csv = os.path.join(base_dir, "pairs.csv")
    
    # Tạo 1000 positive pairs và 1000 negative pairs từ 100% dữ liệu
    generate_pairs_from_labels(
        labels_csv,
        output_csv,
        num_positive_pairs=1000,
        num_negative_pairs=1000
    )
