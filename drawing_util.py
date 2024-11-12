import cv2
import numpy as np
from matplotlib import pyplot as plt

def overlay_heatmap(img, att, cmap=plt.cm.jet):
    gamma = 1.0
    att = cv2.blur(att, (35, 35))
    colorized = cmap(np.uint8(att*255))
    alpha = 0.7
    overlaid = np.uint8(img*(1-alpha)+colorized[:, :, 2::-1]*255*alpha)
    return overlaid

def draw_bounding_box_around_important_regions(image_path, heatmap, output_path, color=(0, 0, 255), thickness=2, top_n=0.1):
    """
    Draw a bounding box around the most important regions of an image based on a heatmap.

    Parameters:
    - image_path: str, path to the input image.
    - heatmap: numpy array, normalized heatmaps.
    - output_path: str, path to save the output image.
    - color: tuple, BGR color of the bounding box.
    - thickness: int, thickness of the bounding box border.
    - top_n: float, top n percents of important regions to include in the bounding box calculation.
    """

    # Load the image
    image = cv2.imread(image_path)

    # Resize heatmap to match image size if necessary
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Find the indices of the top N values in the heatmap
    top_n = int((image.shape[1]*image.shape[0])*top_n)
    flat_heatmap = heatmap.flatten()
    indices = np.argpartition(flat_heatmap, -top_n)[-top_n:]
    indices = indices[np.argsort(-flat_heatmap[indices])]

    # Convert flat indices to 2D coordinates
    coords = np.array(np.unravel_index(indices, heatmap.shape)).T  # Shape: (N, 2)
    # Note: coords are in (row, col) order, i.e., (y, x)

    # Get the bounding rectangle of the points
    x_coords = coords[:, 1]
    y_coords = coords[:, 0]
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)

    # Draw the rectangle on the image
    top_left = (int(x_min), int(y_min))
    bottom_right = (int(x_max), int(y_max))
    cv2.rectangle(image, top_left, bottom_right, color, thickness)

    # Save the output image
    cv2.imwrite(output_path, image)


def draw_optimal_bounding_box(image_path, heatmap, output_path, color=(0, 0, 255), thickness=2, max_area_ratio=0.3):
    """
    Draw a bounding box that maximizes the sum of heatmap values within it,
    ensuring the bounding box area does not exceed a specified ratio of the image area.

    Parameters:
    - image_path: str, path to the input image.
    - heatmap: numpy array, normalized heatmaps.
    - output_path: str, path to save the output image.
    - color: tuple, BGR color of the bounding box.
    - thickness: int, thickness of the bounding box border.
    - max_area_ratio: float, maximum allowable ratio of bounding box area to image area (e.g., 0.3 for 30%).
    """
    # Load the image
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]
    image_area = img_width * img_height
    max_area = max_area_ratio * image_area

    # Resize heatmap to match image size if necessary
    if heatmap.shape != (img_height, img_width):
        heatmap = cv2.resize(heatmap, (img_width, img_height), interpolation=cv2.INTER_LINEAR)

    image = overlay_heatmap(image, heatmap/heatmap.max())

    # Compute the integral image (summed-area table) of the heatmap
    integral_heatmap = cv2.integral(heatmap)

    # Define possible rectangle sizes (heights and widths)
    # We'll sample rectangle sizes to reduce computational load
    heights = [int(img_height * r) for r in np.linspace(0.05, 1.0, num=20)]
    widths = [int(img_width * r) for r in np.linspace(0.05, 1.0, num=20)]

    # Initialize variables to keep track of the best bounding box
    max_sum = -np.inf
    best_rect = None

    # Iterate over possible rectangle sizes
    for h in heights:
        for w in widths:
            # Skip if area exceeds max_area
            if h * w > max_area:
                continue

            # Slide the window over the image
            for y in range(0, img_height - h + 1, max(1, h // 10)):
                for x in range(0, img_width - w + 1, max(1, w // 10)):
                    # Compute the sum of the heatmap within the current rectangle using the integral image
                    sum_val = (
                        integral_heatmap[y + h, x + w]
                        - integral_heatmap[y, x + w]
                        - integral_heatmap[y + h, x]
                        + integral_heatmap[y, x]
                    )

                    # Update the best rectangle if current sum is greater
                    if sum_val > max_sum:
                        max_sum = sum_val
                        best_rect = (x, y, w, h)

    # If a suitable rectangle was found, draw it on the image
    if best_rect is not None:
        x, y, w, h = best_rect
        top_left = (int(x), int(y))
        bottom_right = (int(x + w - 1), int(y + h - 1))
        cv2.rectangle(image, top_left, bottom_right, color, thickness)
    else:
        print("No suitable bounding box found within the area constraint.")

    # Save the output image
    cv2.imwrite(output_path, image)