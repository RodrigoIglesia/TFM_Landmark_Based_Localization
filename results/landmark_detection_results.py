import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Import tensorflow before transformers with logs deactivated to avoid printing tf logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

from PIL import Image
from pycocotools.coco import COCO
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from tqdm import tqdm
import evaluate

# Add project src root to python path
current_script_directory = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.abspath(os.path.join(current_script_directory, ".."))
sys.path.insert(0, src_dir)

target_classes = torch.tensor([5, 6, 7])
NUM_CLASSES = 2


class CocoFilteredMaskDataset(Dataset):
    def __init__(self, image_dir, annotation_file, extractor, target_classes):
        """
        Args:
            image_dir (str): Directory containing images.
            annotation_file (str): Path to COCO-format JSON file.
            extractor: Segformer feature extractor.
            target_classes (torch.Tensor or list[int]): List or tensor of category IDs to keep as foreground (label=1).
        """
        self.image_dir = image_dir
        self.extractor = extractor
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

        # Store target category IDs as a tensor for fast comparison
        self.target_cat_ids = torch.tensor(target_classes) if not isinstance(target_classes, torch.Tensor) else target_classes

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, img_info["file_name"])
        image = Image.open(image_path).convert("RGB")

        height, width = img_info["height"], img_info["width"]
        mask = np.zeros((height, width), dtype=np.uint8)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        for ann in anns:
            if ann["iscrowd"] == 0:
                ann_mask = self.coco.annToMask(ann)
                category_id = ann["category_id"]
                # Apply the category_id to the mask (overwrite existing values)
                mask[ann_mask == 1] = category_id

        encoded = self.extractor(image, return_tensors="pt")

        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": torch.tensor(mask, dtype=torch.long),
            "image": image
        }

    
def load_coco_split(dataset_dir, split_name, feature_extractor):
    image_dir = os.path.join(dataset_dir, split_name)
    annotation_path = os.path.join(dataset_dir, split_name, "_annotations.coco.json")
    dataset = CocoFilteredMaskDataset(image_dir, annotation_path, feature_extractor, target_classes)
    return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)


def predict(model, extractor, image, device):
    pixel_values = extractor(image, return_tensors='pt').pixel_values.to(device)
    with torch.no_grad():
        logits = model(pixel_values).logits
        logits = F.interpolate(
            logits,
            size=image.size[::-1],  # (width, height) → (height, width)
            mode='bilinear',
            align_corners=False
        )
    labels = torch.argmax(logits.squeeze(), dim=0)
    return labels


def evaluate_split(model, loader, split_name, feature_extractor, device):
    metric = evaluate.load("mean_iou")
    for batch in tqdm(loader, desc="Evaluating"):
        sample = batch[0]
        image = sample["image"]
        label = sample["labels"].to(device)

        prediction = predict(model, feature_extractor, image, device)
        binary_prediction = torch.isin(prediction, target_classes).long()
        predictions=binary_prediction.unsqueeze(0).cpu().numpy()
        references=label.unsqueeze(0).numpy()
        # print(f"predictions shape: {predictions.shape}")
        # print(f"references shape: {references.shape}")
        metric.add_batch(
            predictions=predictions,
            references=references
        )
        # print(metric.compute(num_labels=2, ignore_index=None))

        metric.add_batch(
            predictions=predictions,
            references=references
        )

    results = metric.compute(num_labels=NUM_CLASSES, ignore_index=None)
    print(f"\n{split_name.capitalize()} Metrics:")
    for k, v in results.items():
        print(f"{k}: {v}")

## Load segformer model for evaluation
model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
model = SegformerForSemanticSegmentation.from_pretrained(model_name)
feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)

dataset_dir = os.path.join(src_dir, "dataset/segmentation/waymo_landmark_segmentation.v1-v1_poles.coco-segmentation"
)

train_loader = load_coco_split(dataset_dir, "train", feature_extractor)
val_loader = load_coco_split(dataset_dir, "valid", feature_extractor)

# evaluate_split(model, train_loader, "train", feature_extractor, 'cpu')
# evaluate_split(model, val_loader, "valid", feature_extractor, 'cpu')

metric = evaluate.load("mean_iou")
for batch in tqdm(train_loader, desc="Evaluating"):
    sample = batch[0]
    image = sample["image"]
    label = sample["labels"].to('cpu')

    prediction = predict(model, feature_extractor, image, 'cpu')
    binary_prediction = torch.isin(prediction, target_classes).long()
    predictions=binary_prediction.unsqueeze(0).cpu().numpy()
    references=label.unsqueeze(0).numpy()
    print(f"predictions shape: {predictions.shape}")
    print(f"references shape: {references.shape}")
    metric.add_batch(
        predictions=predictions,
        references=references
    )
    print(metric.compute(num_labels=2, ignore_index=None))

    metric.add_batch(
        predictions=predictions,
        references=references
    )

    if torch.is_tensor(label):
        label_mask = label.cpu().numpy()
    if torch.is_tensor(binary_prediction):
        prediction_mask = binary_prediction.cpu().numpy()

    # Scale up binary masks for visibility (0 or 255)
    label_vis = (label_mask * 255).astype(np.uint8)
    prediction_vis = (prediction_mask * 255).astype(np.uint8)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(label_vis, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(prediction_vis, cmap='gray')
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

# results = metric.compute(num_labels=NUM_CLASSES)