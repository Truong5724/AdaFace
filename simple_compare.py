import net
import torch
from face_alignment import align


def load_pretrained_model(model_path):
    """Load AdaFace pretrained model"""
    model = net.build_model('ir_50')
    statedict = torch.load(model_path, map_location='cpu')['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model


def to_input(pil_rgb_image):
    """Convert PIL RGB image to BGR tensor input"""
    import numpy as np
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return tensor


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
    model = load_pretrained_model('models/adaface_ir50_casia.ckpt')
    print("✓ Đã load model thành công!\n")
    
    # So sánh 2 ảnh - THAY ĐỔI ĐƯỜNG DẪN CỦA BẠN Ở ĐÂY
    image1 = 'F:\Biometric\datasets\lfw-deepfunneled\lfw-deepfunneled\Aaron_Peirsol\Aaron_Peirsol_0001.jpg'  # Thay bằng đường dẫn ảnh của bạn
    image2 = 'F:\Biometric\datasets\lfw-deepfunneled\lfw-deepfunneled\Aaron_Guiel\Aaron_Guiel_0001.jpg'  # Thay bằng đường dẫn ảnh của bạn
    
    similarity = compare_two_faces(model, image1, image2)
