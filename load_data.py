from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent
TEST_DATA_ROOT = REPO_ROOT / "test_data"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
DATASET_DIRS = {
    "celeba": {
        "train": Path(r"C:\Roman\datasets\CelebA2\train"),
        "test": Path(r"C:\Roman\datasets\CelebA2\test"),
    },
    "div2k": {
        "train": Path(r"C:\Roman\datasets\DIV2K_patches\patches\train"),
        "test": Path(r"C:\Roman\datasets\DIV2K_patches\patches\valid"),
    },
    "places": {
        "train": Path(r"C:\Roman\NPN_Clean\NPN\data\val_256\train"),
        "test": Path(r"C:\Roman\NPN_Clean\NPN\data\val_256\test"),
    },
    "sar": {
        "train": Path(r"C:\Roman\datasets\SAR_patches\train"),
        "test": Path(r"C:\Roman\datasets\SAR_patches\test"),
    },
}


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, debug=False, num_images=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_paths = self._collect_image_paths(debug=debug, num_images=num_images)
        self.images = self._preload_images()

    def _collect_image_paths(self, debug=False, num_images=None):
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")

        image_paths = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if num_images is not None:
            image_paths = image_paths[:num_images]
        if debug:
            image_paths = image_paths[: min(4, len(image_paths))]
        if not image_paths:
            raise ValueError(f"No images found in {self.image_dir}")
        return image_paths

    def _preload_images(self):
        def load_image(img_path):
            with Image.open(img_path) as image:
                image = image.convert("RGB")
                if self.transform:
                    image = self.transform(image)
            return image

        with ThreadPoolExecutor() as executor:
            return list(
                tqdm(
                    executor.map(load_image, self.image_paths),
                    total=len(self.image_paths),
                    desc=f"Loading {self.image_dir.name}",
                )
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


def build_transform(args):
    transform_steps = [transforms.Resize((args.n, args.n), antialias=True)]
    if args.grayscale:
        transform_steps.append(transforms.Grayscale())
    transform_steps.append(transforms.ToTensor())
    return transforms.Compose(transform_steps)


def resolve_dataset_dir(args, split):
    dataset = args.dataset.lower()
    if dataset not in DATASET_DIRS:
        supported = ", ".join(sorted(DATASET_DIRS))
        raise ValueError(f"Unsupported dataset '{args.dataset}'. Supported datasets: {supported}.")

    if split == "test" and getattr(args, "use_test_data", False):
        test_dir = TEST_DATA_ROOT / dataset
        if not test_dir.exists():
            raise FileNotFoundError(f"Missing repo test data directory: {test_dir}")
        return test_dir

    return DATASET_DIRS[dataset][split]


def build_dataloader(args, split):
    dataset = ImageDataset(
        resolve_dataset_dir(args, split),
        transform=build_transform(args),
        debug=getattr(args, "debug", False),
        num_images=getattr(args, "num_train_images", None) if split == "train" else None,
    )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == "train"))


def get_dataloaders(args):
    return build_dataloader(args, "train"), build_dataloader(args, "test")


def get_test_dataloader(args):
    return build_dataloader(args, "test")
