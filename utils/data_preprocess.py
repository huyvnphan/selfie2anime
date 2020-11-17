import os

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm


def has_file_allowed_extension(filename):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif"]
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def get_file_names(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname):
                path = os.path.join(root, fname)
                item = path
                images.append(item)

    return images


def make_dataset(data_path, split):
    IMAGE_SIZE = 256
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    to_tensor = T.Compose([T.Resize(IMAGE_SIZE), T.ToTensor(), T.Normalize(mean, std)])

    file_names = get_file_names(os.path.join(data_path, split))
    dataset = torch.zeros(
        len(file_names), 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32
    )

    for i, img_path in enumerate(tqdm(file_names)):
        dataset[i] = to_tensor(Image.open(img_path))

    name = os.path.join(data_path, split + ".pt")
    torch.save(dataset, name)
    print("Dataset shape:", dataset.size())
    print("Successfully saved file as:", name)
