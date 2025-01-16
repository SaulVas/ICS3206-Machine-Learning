import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# pylint: disable=all


def compute_similarity(image: np.ndarray, template: np.ndarray) -> float:
    """
    Compute normalized cross-correlation between image and template
    """
    # Ensure both images are the same size
    if image.shape != template.shape:
        template = cv2.resize(template, (image.shape[1], image.shape[0]))

    # Compute normalized cross-correlation
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    return np.max(result)


def evaluate_templates(
    image_paths: List[str],
    template_paths: List[str],
    is_combined: bool = False,
    threshold: float = 0.7,
) -> Dict:
    """
    Evaluate template matching performance and compute metrics.
    For each image:
    - TP: Correctly identified the right template
    - FP: Incorrectly matched to wrong template
    - FN: Failed to match any template (similarity below threshold)
    """

    # Load all templates
    templates = {}
    for template_pattern in template_paths:
        for template_path in glob.glob(template_pattern):
            template_name = Path(template_path).stem
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                print(f"Warning: Could not load template: {template_path}")
                continue
            templates[template_name] = template

    if not templates:
        raise ValueError("No templates could be loaded. Check template paths.")

    overall_metrics = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
    }

    template_similarities = {} if is_combined else None
    if is_combined:
        template_similarities = {
            template_name: [] for template_name in templates.keys()
        }

    # Process each image
    total_images = 0
    for image_pattern in image_paths:
        for image_path in glob.glob(image_pattern):
            total_images += 1
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Could not load image: {image_path}")
                continue

            image_name = Path(image_path).stem
            true_template = next(
                (name for name in templates.keys() if name in image_name), None
            )

            if not true_template:
                print(f"Warning: Could not determine true template for {image_name}")
                continue

            # Track best match for this image
            best_match = {"template": None, "similarity": -1}

            # Find the best matching template
            for template_name, template in templates.items():
                similarity = compute_similarity(image, template)

                # Store similarity for combined dataset
                if is_combined and template_name == true_template:
                    template_similarities[template_name].append(similarity)

                # Track the best match
                if similarity > best_match["similarity"]:
                    best_match = {"template": template_name, "similarity": similarity}

            # Evaluate the best match
            if best_match["similarity"] >= threshold:
                if best_match["template"] == true_template:
                    overall_metrics["true_positives"] += 1
                else:
                    overall_metrics["false_positives"] += 1
            else:
                overall_metrics["false_negatives"] += 1

    # Calculate metrics
    tp = overall_metrics["true_positives"]
    fp = overall_metrics["false_positives"]
    fn = overall_metrics["false_negatives"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    results = {
        "debug_info": {
            "total_images": total_images,
            "total_templates": len(templates),
        },
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        },
    }

    # Add template similarities only for combined dataset
    if is_combined and template_similarities:
        results["template_similarities"] = {
            template_name: sum(similarities) / len(similarities) if similarities else 0
            for template_name, similarities in template_similarities.items()
        }

    print(f"\nDebug Info for {image_paths}:")
    print(f"Total images processed: {total_images}")
    print(f"Total templates: {len(templates)}")
    print(f"Metrics: {overall_metrics}")

    return results


def plot_f1_scores(all_results: Dict) -> None:
    """
    Plot F1 scores for each augmentation technique and level
    """
    techniques = list(all_results["individual_augmentations"].keys())
    levels = [1, 2, 3]

    # Calculate average F1 scores for each technique and level
    f1_scores = {level: [] for level in levels}

    for technique in techniques:
        for level in levels:
            # Get results for this technique and level
            level_results = all_results["individual_augmentations"][technique].get(
                level, {}
            )

            # Calculate average F1 score across all templates
            if isinstance(level_results, dict) and not "error" in level_results:
                f1_scores_level = [
                    template_data["metrics"]["f1_score"]
                    for template_data in level_results.values()
                ]
                avg_f1 = (
                    sum(f1_scores_level) / len(f1_scores_level)
                    if f1_scores_level
                    else 0
                )
                f1_scores[level].append(avg_f1)
            else:
                f1_scores[level].append(0)  # Handle error cases

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot a line for each level
    for level in levels:
        plt.plot(techniques, f1_scores[level], marker="o", label=f"Level {level}")

    plt.xlabel("Augmentation Technique")
    plt.ylabel("Average F1 Score")
    plt.title("Average F1 Scores by Augmentation Technique and Level")
    plt.legend()
    plt.grid(True)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig("f1_scores.png")
    plt.close()


def plot_metric_scores(all_results: Dict, metric_name: str) -> None:
    """
    Plot specified metric scores for each augmentation technique and level
    """
    techniques = list(all_results["individual_augmentations"].keys())
    levels = [1, 2, 3]

    # Calculate average scores for each technique and level
    scores = {level: [] for level in levels}

    for technique in techniques:
        for level in levels:
            level_results = all_results["individual_augmentations"][technique].get(
                level, {}
            )

            # Check if level_results is a dictionary and contains valid metrics
            if (
                isinstance(level_results, dict)
                and not "error" in level_results
                and "metrics" in level_results
            ):  # Add this check
                metric_scores = [level_results["metrics"][metric_name]]
                avg_score = (
                    sum(metric_scores) / len(metric_scores) if metric_scores else 0
                )
                scores[level].append(avg_score)
            else:
                scores[level].append(0)

    plt.figure(figsize=(12, 6))

    for level in levels:
        plt.plot(techniques, scores[level], marker="o", label=f"Level {level}")

    plt.xlabel("Augmentation Technique")
    plt.ylabel(f'Average {metric_name.replace("_", " ").title()}')
    plt.title(
        f'Average {metric_name.replace("_", " ").title()} by Augmentation Technique and Level'
    )
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(f"{metric_name}_scores.png")
    plt.close()


def plot_template_similarities(all_results: Dict) -> None:
    """
    Plot average similarities for each template across combined augmentations
    """
    if "combined_augmentations" not in all_results:
        print("No combined augmentation results found")
        return

    combined_results = all_results["combined_augmentations"]
    if isinstance(combined_results, dict) and not "error" in combined_results:
        templates = list(combined_results["template_similarities"].keys())
        similarities = [
            combined_results["template_similarities"][template]
            for template in templates
        ]

        plt.figure(figsize=(12, 6))
        plt.bar(templates, similarities)

        plt.xlabel("Template Name")
        plt.ylabel("Average Similarity")
        plt.title("Average Template Similarities in Combined Augmentations")
        plt.grid(True, axis="y")
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig("template_similarities.png")
        plt.close()


def evaluate_with_confusion_matrix(template_paths: List[str]) -> Dict:
    """
    Evaluate template matching performance and create a confusion matrix
    """
    # Load templates
    templates = {}
    for template_pattern in template_paths:
        for template_path in glob.glob(template_pattern):
            template_name = Path(template_path).stem
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                print(f"Warning: Could not load template: {template_path}")
                continue
            templates[template_name] = template

    if not templates:
        raise ValueError("No templates could be loaded. Check template paths.")

    # Initialize lists to store true and predicted labels
    y_true = []
    y_pred = []
    template_names = list(templates.keys())

    # Process all images (both individual and combined)
    dataset_patterns = [
        "./data/augmented/individual/*.jpg",
        "./data/augmented/combined/*.jpg",
    ]

    for pattern in dataset_patterns:
        for image_path in glob.glob(pattern):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Could not load image: {image_path}")
                continue

            image_name = Path(image_path).stem
            true_template = next(
                (name for name in templates.keys() if name in image_name), None
            )

            if not true_template:
                print(f"Warning: Could not determine true template for {image_name}")
                continue

            # Find best matching template
            best_match = {"template": None, "similarity": -1}
            for template_name, template in templates.items():
                similarity = compute_similarity(image, template)
                if similarity > best_match["similarity"]:
                    best_match = {"template": template_name, "similarity": similarity}

            y_true.append(true_template)
            y_pred.append(best_match["template"])

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=template_names)

    # Calculate metrics
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    # Create confusion matrix plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=template_names,
        yticklabels=template_names,
    )
    plt.title("Template Matching Confusion Matrix")
    plt.xlabel("Predicted Template")
    plt.ylabel("True Template")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Print metrics
    print("\nOverall Metrics:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"\nTotal images processed: {len(y_true)}")

    return {
        "confusion_matrix": cm,
        "metrics": {"precision": precision, "recall": recall, "f1_score": f1},
        "total_images": len(y_true),
    }


# Example usage:
if __name__ == "__main__":
    data_dir = Path("data")
    template_paths = ["./data/original/*.png"]

    print("\nRunning comprehensive evaluation with confusion matrix...")
    results = evaluate_with_confusion_matrix(template_paths)
