import argparse
import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2  
from models.x1 import X1
from models.x2 import X2
from models.x3 import X3
from models.x4 import X4

def load_model(weight_path, model_type):
    if model_type == 'x1':
        model = X1(num_classes=4).cuda()
    elif model_type == 'x2':
        model = X2(num_classes=4).cuda()
    elif model_type == 'x3':
        model = X3(num_classes=4).cuda()
    elif model_type == 'x4':
        model = X4(num_classes=4).cuda()
    else:
        raise ValueError("Unsupported model type. Choose either 'x1', 'x2', 'x3' or 'x4'.")
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),  # Adjust this to match your input size
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('L')
    input_tensor = transform(image).unsqueeze(0).cuda()  # Add batch dimension and move to GPU
    return input_tensor

def predict(image_path, weight_path, model_type):
    model = load_model(weight_path, model_type)
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)  # output: [1, 4, 1024, 1024]
        if isinstance(output, tuple):
            output = output[0]
        prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # shape: [H, W]
    return prediction

def save_prediction(prediction, output_path):
    # Save class index map (grayscale where each pixel is class label)
    Image.fromarray(prediction.astype(np.uint8)).save(output_path)

def load_ground_truth(gt_path):
    gt = Image.open(gt_path).convert('L').resize((1024, 1024), resample=Image.NEAREST)
    return np.array(gt).astype(np.uint8)

def compute_iou(prediction, ground_truth, num_classes=4):
    """
    Computes IoU for each class individually.
    prediction: H x W (predicted class labels)
    ground_truth: H x W (true class labels)
    """
    ious = []
    for cls in range(num_classes):
        pred_cls = (prediction == cls).astype(np.uint8)
        gt_cls = (ground_truth == cls).astype(np.uint8)

        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()

        if union == 0:
            ious.append(float('nan'))  # To show it's undefined for this class
        else:
            ious.append(intersection / union)
    return ious

def main():
    parser = argparse.ArgumentParser(description='Predict with a trained model, compute IoU and overlay mask edges')
    parser.add_argument('--weight', required=True, type=str, help='Path to the model weight file')
    parser.add_argument('--image', required=True, type=str, help='Path to the input folder')
    parser.add_argument('--model', required=True, type=str, choices=['x1', 'x2', 'x3', 'x4'], help='Model type')
    parser.add_argument('--gt', type=str, default=None, help='Path to the ground truth folder (optional, for IoU calculation)')
    args = parser.parse_args()
        
    # Create output directories if they don't exist
    output_dir = 'predict/exp16'  #directory path to save predictions
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = sorted(os.listdir(args.image))
    total_inference_time = 0
    count = 0
    
    for image_file in image_files:
        image_path = os.path.join(args.image, image_file)
        
        start_time = time.time()
        prediction = predict(image_path, args.weight, args.model)
        inference_time = time.time() - start_time
        total_inference_time += inference_time
        count += 1
        
        print(f"Inference time for {image_file}: {inference_time:.8f} seconds")
        
        if args.gt is not None:
            gt_path = os.path.join(args.gt, image_file)
            if os.path.exists(gt_path):
                ground_truth = load_ground_truth(gt_path)
                ious = compute_iou(prediction, ground_truth, num_classes=4)
                print(f"IoUs for {image_file}:")
                for cls, iou in enumerate(ious):
                    if np.isnan(iou):
                        print(f" - Class {cls}: IoU undefined (no ground truth or prediction)")
                    else:
                        print(f" - Class {cls}: {iou:.4f}")
        
        # Save the prediction
        output_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}.png")
        save_prediction(prediction, output_path)
        print(f"Saved prediction for {image_file} to {output_path}")
    
    if count > 0:
        avg_time = total_inference_time / count
        print(f"Average inference time per image: {avg_time:.8f} seconds")

if __name__ == "__main__":
    main()
