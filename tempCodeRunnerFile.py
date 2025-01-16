from pathlib import Path
import cv2
import numpy as np
from typing import Dict
from skimage.util import random_noise
from skimage import color

# pylint: disable=all


def create_combined_augmentation(
    image: np.ndarray, augmentation_funcs: Dict
) -> tuple[np.ndarray, str]:
    """
    Creates a combined augmentation using 3 random techniques with random levels.
    Returns the augmented image and a descriptive name of the augmentations applied.
    """
    # List of available augmentation types
    aug_types = [
        "shift",
        "contrast",
        "noise",
        "blur",
        "rotation",
        "brightness",
        "jet",
    ]

    # Randomly select 3 unique augmentation types
    selected_types = np.random.choice(aug_types, size=3, replace=False)

    # Start with the original image
    result = image.copy()

    # Build the name and apply augmentations
    name_parts = []
    for aug_type in selected_types:
        # Random intensity level (1, 2, or 3)
        level = np.random.randint(1, 4)
        key = f"{aug_type}_{level}"
        result = augmentation_funcs[key](result)
        name_parts.append(f"{aug_type}_{level}")

    # Create the full name
    full_name = "combined_" + "_".join(name_parts)

    return result, full_name


def apply_augmentations(image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Applies various augmentation techniques at three intensity levels.
    Level 1: Very subtle changes
    Level 2: Moderate changes
    Level 3: Aggressive changes

    Args:
        image (np.ndarray): Input image in BGR format

    Returns:
        Dict[str, np.ndarray]: Dictionary containing original and augmented images
    """
    augmented_images = {"original": image.copy()}

    # Contrast adjustment
    # Level 1: Barely noticeable, Level 2: Moderate, Level 3: Strong
    contrast_factors = [1.05, 1.5, 2.0]
    for i, factor in enumerate(contrast_factors, 1):
        augmented_images[f"contrast_{i}"] = cv2.convertScaleAbs(
            image, alpha=factor, beta=0
        )

    # Brightness adjustment
    # Level 1: Slight brightening, Level 2: Noticeable, Level 3: Significant
    brightness_factors = [10, 50, 70]
    for i, factor in enumerate(brightness_factors, 1):
        augmented_images[f"brightness_{i}"] = cv2.convertScaleAbs(
            image, alpha=1, beta=factor
        )

    # Add Gaussian noise
    # Level 1: Very fine grain, Level 2: Moderate noise, Level 3: Heavy noise
    def add_noise(img, var):
        # random_noise expects float image and returns float image
        float_img = img.astype("float32") / 255.0
        noisy_float = random_noise(float_img, mode="gaussian", var=var)
        # Convert back to uint8
        return (noisy_float * 255).astype(np.uint8)

    noise_levels = [0.001, 0.01, 0.03]  # Variance levels for gaussian noise
    for i, level in enumerate(noise_levels, 1):
        augmented_images[f"noise_{i}"] = add_noise(image, level)

    # Blur
    # Level 1: Slight softening, Level 2: Noticeable blur, Level 3: Strong blur
    blur_kernels = [
        (7, 7),
        (11, 11),
        (21, 21),
    ]
    for i, kernel in enumerate(blur_kernels, 1):
        sigma = 0.5 if i == 1 else 0  # Smaller sigma for level 1
        augmented_images[f"blur_{i}"] = cv2.GaussianBlur(image, kernel, sigma)

    # Rotation
    # Level 1: Very subtle tilt, Level 2: Slight tilt, Level 3: Moderate rotation
    def rotate_image(img, angle):
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, rotation_matrix, (width, height))

    rotation_angles = [2, 5, 10]
    for i, angle in enumerate(rotation_angles, 1):
        augmented_images[f"rotation_{i}"] = rotate_image(image, angle)

    # Scale/Shift manipulation
    # Level 1: Slight shift, Level 2: Moderate shift, Level 3: Larger shift
    def shift_image(img, shift_x, shift_y):
        height, width = img.shape[:2]
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        return cv2.warpAffine(
            img, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )  # Black border

    # Shift by percentage of image size
    shift_factors = [0.02, 0.05, 0.1]  # 2%, 5%, 10% of image size
    for i, factor in enumerate(shift_factors, 1):
        height, width = image.shape[:2]
        shift_x = int(width * factor)
        shift_y = int(height * factor)
        # Random direction for each shift
        shift_x *= np.random.choice([-1, 1])
        shift_y *= np.random.choice([-1, 1])
        augmented_images[f"shift_{i}"] = shift_image(image, shift_x, shift_y)

    # Color jet effect with different intensity levels
    # Level 1: Subtle blend, Level 2: Medium blend, Level 3: Strong blend
    def apply_jet(img, alpha):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        return cv2.addWeighted(img, 1 - alpha, colored, alpha, 0)

    jet_intensities = [0.1, 0.6, 0.9]
    for i, intensity in enumerate(jet_intensities, 1):
        augmented_images[f"jet_{i}"] = apply_jet(image, intensity)

    augmentation_funcs = {
        "contrast_1": lambda img: cv2.convertScaleAbs(img, alpha=1.05, beta=0),
        "contrast_2": lambda img: cv2.convertScaleAbs(img, alpha=1.5, beta=0),
        "contrast_3": lambda img: cv2.convertScaleAbs(img, alpha=2.0, beta=0),
        "brightness_1": lambda img: cv2.convertScaleAbs(img, alpha=1, beta=10),
        "brightness_2": lambda img: cv2.convertScaleAbs(img, alpha=1, beta=50),
        "brightness_3": lambda img: cv2.convertScaleAbs(img, alpha=1, beta=70),
        "noise_1": lambda img: add_noise(img, 0.001),
        "noise_2": lambda img: add_noise(img, 0.01),
        "noise_3": lambda img: add_noise(img, 0.03),
        "blur_1": lambda img: cv2.GaussianBlur(img, (7, 7), 0.5),
        "blur_2": lambda img: cv2.GaussianBlur(img, (11, 11), 0),
        "blur_3": lambda img: cv2.GaussianBlur(img, (21, 21), 0),
        "rotation_1": lambda img: rotate_image(img, 2),
        "rotation_2": lambda img: rotate_image(img, 5),
        "rotation_3": lambda img: rotate_image(img, 30),
        "shift_1": lambda img: shift_image(
            img,
            int(img.shape[1] * 0.02) * np.random.choice([-1, 1]),
            int(img.shape[0] * 0.02) * np.random.choice([-1, 1]),
        ),
        "shift_2": lambda img: shift_image(
            img,
            int(img.shape[1] * 0.05) * np.random.choice([-1, 1]),
            int(img.shape[0] * 0.05) * np.random.choice([-1, 1]),
        ),
        "shift_3": lambda img: shift_image(
            img,
            int(img.shape[1] * 0.1) * np.random.choice([-1, 1]),
            int(img.shape[0] * 0.1) * np.random.choice([-1, 1]),
        ),
        "jet_1": lambda img: apply_jet(img, 0.1),
        "jet_2": lambda img: apply_jet(img, 0.6),
        "jet_3": lambda img: apply_jet(img, 0.9),
    }

    # Create 5 random combined augmentations
    for i in range(5):
        combined_img, name = create_combined_augmentation(image, augmentation_funcs)
        augmented_images[name] = combined_img
        # Optionally log the combination created
        print(f"Created combination {i+1}: {name}")

    return augmented_images


if __name__ == "__main__":
    # Create separate directories for individual and combined augmentations
    base_output_dir = Path("data/augmented")
    individual_dir = base_output_dir / "individual"
    combined_dir = base_output_dir / "combined"

    # Create directories
    individual_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)

    # Get input directory
    input_dir = Path("data/original")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist")

    # Process each image
    for image_path in input_dir.glob("*.png"):
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                continue

            # Apply augmentations
            augmented_images = apply_augmentations(image)

            # Save augmented images
            for key, aug_image in augmented_images.items():
                # Determine output directory based on augmentation type
                if key.startswith("combined_"):
                    output_path = combined_dir / f"{image_path.stem}_{key}.jpg"
                else:
                    output_path = individual_dir / f"{image_path.stem}_{key}.jpg"

                cv2.imwrite(str(output_path), aug_image)

            print(f"Successfully processed: {image_path.name}")

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
