import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path


class ResNet18FeatureExtractor(nn.Module):
    """
    ResNet-18 backbone that outputs 512-D feature embeddings.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        backbone = models.resnet18(pretrained=pretrained)

        # Remove classification head
        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-1]  # removes FC layer
        )

        self.output_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 3, 224, 224)

        Returns:
            Tensor of shape (B, 512)
        """
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return features


# Standard transforms (shared across training & inference)
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extract_features_from_images(
    image_paths,
    model: ResNet18FeatureExtractor,
    device: str = "cuda"
):
    """
    Extract features for a list of image paths.

    Args:
        image_paths (List[str | Path])
        model (ResNet18FeatureExtractor)
        device (str)

    Returns:
        Tensor of shape (N, 512)
    """

    model.eval()
    model.to(device)

    features = []

    with torch.no_grad():
        for img_path in image_paths:
            img_path = Path(img_path)
            image = Image.open(img_path).convert("RGB")
            tensor = image_transform(image).unsqueeze(0).to(device)
            feat = model(tensor)
            features.append(feat.cpu())

    return torch.cat(features, dim=0)
