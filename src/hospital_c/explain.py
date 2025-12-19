import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from .model import get_model

# Labels map
LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", 
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", 
    "Consolidation", "Edema", "Emphysema", "Fibrosis", 
    "Pleural_Thickening", "Hernia"
]

def run_gradcam_top3(model_path, img_path, output_dir, num_classes=14):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = model.to(device)
    target_layers = [model.layer4[-1]]

    # Prepare Image
    rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
    rgb_img_float = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img_float, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)

    # Get Predictions
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output).squeeze()
        
    top3_probs, top3_indices = torch.topk(probs, 3)
    
    heatmap_images = []
    titles = []

    cam = GradCAM(model=model, target_layers=target_layers)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate CAM for Top 3
    for i in range(3):
        idx = top3_indices[i].item()
        score = top3_probs[i].item()
        label_name = LABELS[idx]
        
        targets = [ClassifierOutputTarget(idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
        
        # Save individual
        save_path = os.path.join(output_dir, f"heatmap_{i+1}_{label_name}.png")
        cv2.imwrite(save_path, visualization[:, :, ::-1])
        
        heatmap_images.append(visualization)
        titles.append(f"{label_name}: {score:.2f}")

    # Create Montage
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original
    axes[0].imshow(rgb_img)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Heatmaps
    for i in range(3):
        axes[i+1].imshow(heatmap_images[i])
        axes[i+1].set_title(titles[i])
        axes[i+1].axis('off')
        
    montage_path = os.path.join(output_dir, "hospital3_cam_top3.png")
    plt.tight_layout()
    plt.savefig(montage_path)
    plt.close()
    
    print(f"Saved Top-3 heatmaps and montage to {output_dir}")

if __name__ == "__main__":
    MODEL_PATH = r"c:\Users\aishw\codered5\ml\models\hospital3_model.pth"
    DATA_DIR = r"c:\Users\aishw\codered5\data\hospital_c\images"
    
    # Pick random image 
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.jpg') or f.endswith('.png')]
    if files:
        IMG_PATH = os.path.join(DATA_DIR, files[0])
        # We also need a target directory
        OUTPUT_DIR = r"c:\Users\aishw\codered5\ml\shap"
        run_gradcam_top3(MODEL_PATH, IMG_PATH, OUTPUT_DIR)
