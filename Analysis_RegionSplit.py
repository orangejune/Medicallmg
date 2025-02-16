import cv2
import numpy as np
import matplotlib.pyplot as plt

class RegionDecomposer:
    def __init__(self, image, mask):
        self.image = image
        self.mask  = mask


    def split_image_with_mask(self, image, mask, min_size=32, variance_threshold=500):
        """
        Decomposes an image based on pixel intensity variance and mask guidance.

        Args:
            image (numpy.ndarray): Grayscale input image.
            mask (numpy.ndarray): Binary mask highlighting important regions (same size as image).
            min_size (int): Minimum region size before stopping the split.
            variance_threshold (float): Variance threshold for further splitting.

        Returns:
            regions (list): List of image patches with bounding box coordinates.
        """
        h, w = image.shape
        regions = []

        def recursive_split(x, y, width, height):
            # Extract region
            sub_img = image[y:y + height, x:x + width]
            sub_mask = mask[y:y + height, x:x + width]

            # Compute intensity variance
            variance = np.var(sub_img)

            # Check if the mask is present in this region
            mask_presence = np.count_nonzero(sub_mask) > 0

            # If the region is small enough or has low variance AND no mask feature, stop splitting
            if width <= min_size or height <= min_size or (variance < variance_threshold and not mask_presence):
                regions.append(((x, y, width, height), sub_img))
                return

            # Split into 4 quadrants (Quadtree-like)
            half_w, half_h = width // 2, height // 2

            recursive_split(x, y, half_w, half_h)
            recursive_split(x + half_w, y, half_w, half_h)
            recursive_split(x, y + half_h, half_w, half_h)
            recursive_split(x + half_w, y + half_h, half_w, half_h)

        # Start recursive splitting from the full image
        recursive_split(0, 0, w, h)

        return regions


    def generate_mask(self, image, line_positions):
        """
        Creates a binary mask with two lines for guiding decomposition.

        Args:
            image (numpy.ndarray): Grayscale input image.
            line_positions (list of tuples): List of (x1, y1, x2, y2) coordinates for lines.

        Returns:
            mask (numpy.ndarray): Binary mask.
        """
        mask = np.zeros_like(image, dtype=np.uint8)
        for (x1, y1, x2, y2) in line_positions:
            cv2.line(mask, (x1, y1), (x2, y2), 255, 2)  # Draw white lines on black mask
        return mask


    def visualize_regions(self, image, regions):
        """
        Visualizes the decomposed image regions.

        Args:
            image (numpy.ndarray): Original grayscale image.
            regions (list): List of (bounding_box, sub_image) tuples.
        """
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for (x, y, w, h), _ in regions:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)

        plt.figure(figsize=(8, 6))
        plt.imshow(output)
        plt.title("Image Decomposition with Mask Guidance")
        plt.show()


# Example usage
image = ""

# Generate the mask
mask = ""

rd = RegionDecomposer(image, mask)

# Decompose the image using both intensity and mask presence
regions = rd. split_image_with_mask(image, mask, min_size=32, variance_threshold=500)

# Visualize the decomposition
rd. visualize_regions(image, regions)
