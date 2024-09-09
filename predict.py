import argparse
import torch
from torch import nn
from torchvision import models
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt


def get_input_args():
    parser = argparse.ArgumentParser(description='Predict flower name from an image.')
    parser.add_argument('image_path', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--category_names', type=str, default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true')
    return parser.parse_args()


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image_path):
    image = Image.open(image_path)

    if image.size[0] < image.size[1]:
        image.thumbnail((256, 256 * image.size[1] // image.size[0]))
    else:
        image.thumbnail((256 * image.size[0] // image.size[1], 256))

    left = (image.width - 224) / 2
    top = (image.height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))

    np_image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))

    return torch.tensor(np_image).float()

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    image = image.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def predict(image_path, model, topk=5):
    image = process_image(image_path)

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    model.to(device)

    image = image.unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        output = model(image)

    probabilities = torch.nn.functional.softmax(output, dim=1)

    top_probs, top_labels = probabilities.topk(topk, dim=1)

    top_probs = top_probs.cpu().numpy().squeeze()
    top_labels = top_labels.cpu().numpy().squeeze()

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}

    top_classes = [idx_to_class[label] for label in top_labels]

    return top_probs, top_classes

if __name__ == "__main__":
    args = get_input_args()

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)
    image_path = args.image_path
    probs, classes = predict(image_path, model, topk=args.top_k)

    print("Predicted probabilities: ", probs)
    print("Predicted classes: ", [cat_to_name.get(cls, cls) for cls in classes])
