import net
import torch
from face_alignment import align
from inference import load_pretrained_model, to_input

adaface_models = {
    'ir_50':"pretrained/adaface_ir50_casia.ckpt",
}

def compare_two_faces(model, image_path1, image_path2, threshold=0.6):
    """So sánh 2 khuôn mặt
    
    Args:
        model: AdaFace model
        image_path1: Đường dẫn ảnh 1
        image_path2: Đường dẫn ảnh 2
        threshold: Ngưỡng để xác định cùng người (mặc định 0.6)
    
    Returns:
        similarity score (float)
    """
    print(f"\n→ Đang xử lý: {image_path1}")
    aligned_img1 = align.get_aligned_face(image_path1)
    
    print(f"→ Đang xử lý: {image_path2}")
    aligned_img2 = align.get_aligned_face(image_path2)
    
    if aligned_img1 is None or aligned_img2 is None:
        print("✗ Không tìm thấy khuôn mặt!")
        return None
    
    # Extract features
    with torch.no_grad():
        feature1, norm1 = model(to_input(aligned_img1))
        feature2, norm2 = model(to_input(aligned_img2))
    
    # Calculate similarity
    similarity = torch.nn.functional.cosine_similarity(feature1, feature2).item()
    
    print(f"\n{'='*60}")
    print(f"KẾT QUẢ:")
    print(f"  Cosine Similarity: {similarity:.4f}")
    print(f"  Độ tương đồng: {similarity * 100:.2f}%")
    
    if similarity > threshold:
        print(f"  ✓ CÓ THỂ là cùng một người (similarity > {threshold})")
    else:
        print(f"  ✗ KHÁC người (similarity <= {threshold})")
    print(f"{'='*60}\n")
    
    return similarity


# ===== SỬ DỤNG =====
if __name__ == '__main__':
    # Load model
    # Hoặc đường dẫn pretrained/adaface_ir50_casia.ckpt
    model = load_pretrained_model('ir_50')
    print("✓ Đã load model thành công!\n")
    
    # So sánh 2 ảnh - THAY ĐỔI ĐƯỜNG DẪN CỦA BẠN Ở ĐÂY
    image1 = 'face_alignment/test_images/img1.jpeg'  # Thay bằng đường dẫn ảnh của bạn
    image2 = 'face_alignment/test_images/img2.jpeg'  # Thay bằng đường dẫn ảnh của bạn
    
    similarity = compare_two_faces(model, image1, image2)
