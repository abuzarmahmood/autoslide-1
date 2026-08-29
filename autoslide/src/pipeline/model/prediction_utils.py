"""
Common utilities for model prediction and evaluation.

This module provides shared functionality for:
- Model loading and initialization
- Single image prediction
- Common imports and setup
"""

import os
import time
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from autoslide.src import config

# Get directories from config - use project-level artifacts directory
# Navigate from autoslide/src/pipeline/model/ to autoslide/artifacts/
artifacts_dir = os.path.join(
    os.path.dirname(__file__),  # autoslide/src/pipeline/model/
    '..', '..', '..', '..',      # up to project root
    'autoslide', 'artifacts'     # down to autoslide/artifacts/
)
artifacts_dir = os.path.abspath(artifacts_dir)


def initialize_model():
    """
    Initialize and configure the Mask R-CNN model.

    Creates a Mask R-CNN model with ResNet-50 backbone and FPN,
    and configures it for binary segmentation (background and vessel).

    Returns:
        torchvision.models.detection.MaskRCNN: Configured Mask R-CNN model
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn()

    # Configure for binary classification (background + vessel)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes)

    return model


def load_model(model_path=None, device=None):
    """
    Load the trained Mask R-CNN model.

    Args:
        model_path (str): Path to the saved model. If None, uses default path.
        device (torch.device): Device to load model on. If None, auto-detects.

    Returns:
        tuple: (model, device, transform) - Loaded model, device, and transform
    """
    if model_path is None:
        model_path = os.path.join(
            artifacts_dir, 'best_val_mask_rcnn_model.pth')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model structure
    model = initialize_model()

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    transform = T.ToTensor()

    return model, device, transform


def predict_single_image(model, image, device, transform, return_time=False):
    """
    Perform prediction on a single image.

    Args:
        model (torch.nn.Module): Trained Mask R-CNN model
        image (PIL.Image or str): Input image or path to image
        device (torch.device): Device to run inference on
        transform (callable): Image transformation function
        return_time (bool): Whether to return prediction time

    Returns:
        numpy.ndarray or tuple: Combined predicted mask, optionally with prediction time


    TODO: Set threshold for minimum confidence score for masks to accept
    """
    # Handle both PIL Image and file path inputs
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        raise ValueError("Image must be PIL.Image or file path string")

    # Transform image
    img_tensor = transform(image).to(device)

    # Measure prediction time if requested. CUDA kernel launches are async,
    # so an explicit sync is needed on both ends of the timed region for an
    # accurate wall-clock measurement.
    if return_time and device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time() if return_time else None

    with torch.no_grad():
        predictions = model([img_tensor])

    if return_time and device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time() if return_time else None
    prediction_time = (end_time - start_time) if return_time else None

    # Combine all predicted masks
    pred = predictions[0]
    if len(pred["masks"]) > 0:
        # Combine all masks with confidence weighting
        masks = pred["masks"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()

        # Weight masks by their confidence scores
        combined_mask = np.zeros_like(masks[0, 0])
        total_weight = 0

        for mask, score in zip(masks, scores):
            combined_mask += mask[0] * score
            total_weight += score

        # Normalize to 0-1 range, then convert to 0-255
        if total_weight > 0:
            combined_mask = combined_mask / total_weight

        if combined_mask.max() > 0:
            combined_mask = combined_mask / combined_mask.max()

        # Convert to uint8 (0-255 range)
        combined_mask = (combined_mask * 255).astype(np.uint8)
    else:
        # No predictions - create empty mask
        img_array = np.array(image)
        combined_mask = np.zeros(
            (img_array.shape[0], img_array.shape[1]), dtype=np.uint8)

    if return_time:
        return prediction_time, combined_mask
    else:
        return combined_mask


def predict_batch(model, images, device, transform, synchronize_cuda=True):
    """
    Time a single batched forward pass. Timing-only -- does not decode masks.

    Args:
        model (torch.nn.Module): Trained Mask R-CNN model
        images (list[PIL.Image]): Batch of input images
        device (torch.device): Device to run inference on
        transform (callable): Image transformation function
        synchronize_cuda (bool): Call torch.cuda.synchronize() around the
            timed region on CUDA devices, so async kernel launches don't
            leak outside the measured window.

    Returns:
        float: wall-clock seconds for the batched forward pass
    """
    img_tensors = [transform(image).to(device) for image in images]

    if device.type == "cuda" and synchronize_cuda:
        torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        _ = model(img_tensors)

    if device.type == "cuda" and synchronize_cuda:
        torch.cuda.synchronize()
    end_time = time.time()

    return end_time - start_time


def setup_device(force_cpu=False):
    """
    Setup and return the appropriate device for inference.

    Args:
        force_cpu (bool): If True, always return the CPU device regardless
            of CUDA availability.

    Returns:
        torch.device: Device to use for inference
    """
    if force_cpu:
        return torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
