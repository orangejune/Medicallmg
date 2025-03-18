import cv2
import numpy as np
import matplotlib.pyplot as plt
from Analysis_ReadImg import Analysis_ReadImg
from Analysis_MaskColor import MaskBufferExtractor


class AdaptiveRegionDecomposition:
    def __init__(self, image, buffered_mask, min_size=50, intensity_threshold=20):
        """
        Adaptive Region Decomposition using recursive region splitting based on pixel intensity.

        Optimized to start within the bounding box of the buffered mask to speed up processing.

        Args:
            image (numpy.ndarray): Full grayscale image.
            buffered_mask (numpy.ndarray): Buffered mask.
            min_size (int): Minimum allowed size for any split region. If smaller, stop splitting.
            intensity_threshold (int): Stop splitting if variation of intensity is smaller than it
        """
        self.image = image
        self.mask = buffered_mask

        # Get the bounding box of the buffered mask to start splitting within that region
        self.bounding_box = self._find_mask_bounding_box()

        self.min_size = min_size  # Stop splitting when region is too small
        self.intensity_threshold = intensity_threshold
        self.regions = []

    def _find_mask_bounding_box(self):
        """Find the bounding box around the buffered mask to start splitting within that region."""
        if self.mask is None:
            return (0, 0, self.image.shape[1], self.image.shape[0])  # Full image as fallback

        # Find contours of the mask to determine bounding box
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (0, 0, self.image.shape[1], self.image.shape[0])  # Fallback to full image

        x, y, w, h = cv2.boundingRect(np.vstack(contours))  # Get the bounding box
        return (x-20, y-20, x + w +20, y + h +20)  # Return bounding box as (x1, y1, x2, y2) (more larger)

    def split_region(self, x1, y1, x2, y2):
        """
        Recursively split regions based on pixel intensity variance.

        Stops splitting if the region size is smaller than `self.min_size`.

        Args:
            x1, y1: Top-left corner of the region.
            x2, y2: Bottom-right corner of the region.
        """
        width = x2 - x1
        height = y2 - y1

        # Stop if the region is too small
        if width < self.min_size or height < self.min_size:
            self.regions.append((x1, y1, x2, y2))  # Keep the small region
            return

        region = self.image[y1:y2, x1:x2]

        # Check intensity variance
        min_intensity = np.min(region)
        max_intensity = np.max(region)
        intensity_diff = max_intensity - min_intensity

        # Stop splitting if variation is small or region is below min size
        if intensity_diff < self.intensity_threshold:
            self.regions.append((x1, y1, x2, y2))  # Store rectangular region
            return

        # Choose split direction (horizontal or vertical)
        if width > height:  # Wider region, split vertically
            mid_x = (x1 + x2) // 2
            self.split_region(x1, y1, mid_x, y2)
            self.split_region(mid_x, y1, x2, y2)
        else:  # Taller region, split horizontally
            mid_y = (y1 + y2) // 2
            self.split_region(x1, y1, x2, mid_y)
            self.split_region(x1, mid_y, x2, y2)


    def region_decomposition(self):
        """Start recursive region splitting within the mask bounding box."""
        x1, y1, x2, y2 = self.bounding_box
        self.split_region(x1, y1, x2, y2)

    def refine_regions_with_mask(self):
        """Keep regions that intersect with the mask, even if not fully inside."""
        if self.mask is None:
            return

        new_regions = []
        for (x1, y1, x2, y2) in self.regions:
            region_mask = self.mask[y1:y2, x1:x2]
            if np.count_nonzero(region_mask) > 0:  # Keep if any pixel overlaps
                new_regions.append((x1, y1, x2, y2))

        self.regions = new_regions

    def visualize_regions(self):
        """Overlay decomposed rectangular regions on the original image."""
        output_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

        for (x1, y1, x2, y2) in self.regions:
            color = tuple(np.random.randint(0, 255, 3).tolist())  # Random color
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

        plt.figure(figsize=(10, 6))
        plt.imshow(output_image)
        plt.title("Optimized Adaptive Region Decomposition (Starts in Mask Bounding Box)")
        plt.axis("off")
        plt.show()

    def run(self):
        """Run the full region decomposition process."""
        self.region_decomposition()
        if self.mask is not None:
            self.refine_regions_with_mask()
        self.visualize_regions()

    def output_regions(self):
        """Run the full region decomposition process."""
        self.region_decomposition()
        if self.mask is not None:
            self.refine_regions_with_mask()
        # self.visualize_regions()
        return self.regions




if __name__ == "__main__":
    # Example Usage
    # full_image_path = r"KD/New/KD-2-chentianxiang/KD-2-chentianxiang/2-8LAD原始.dcm"
    # blue_image_path = r"KD/New/KD-2-chentianxiang/KD-2-chentianxiang/2-10LAD描记.dcm"
    full_image_path = r"KD/New/KD-3-dengjinyi/KD-3-dengjinyi/3-4LAD原始.dcm"  # Replace with your grayscale image path
    blue_image_path = r"KD/New/KD-3-dengjinyi/KD-3-dengjinyi/3-5LAD描记.dcm"  # Replace with your image with blue lines

    hsv_ranges = {
        "red": [(np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([160, 100, 100]), np.array([179, 255, 255]))]
    }
    target_bbox = (160, 70, 910, 500)
    buffer_size = 10
    min_region_size = 500
    min_size = 120  # Stop splitting if region is smaller than this size

    # Load images using Analysis_ReadImg
    readimg = Analysis_ReadImg(full_image_path, blue_image_path, is_dicom=True)

    image = readimg.full_image  # Grayscale full image
    blue_image = readimg.blue_image  # Image with labeled regions

    # Convert to grayscale if it's a 3-channel image
    if len(image.shape) == 3:
        image = cv2.cvtColor(readimg.full_image, cv2.COLOR_BGR2GRAY)

    # Extract masks using MaskBufferExtractor
    extractor = MaskBufferExtractor(readimg.full_image, readimg.blue_image)
    for color, ranges in hsv_ranges.items():
        print(f"Processing color: {color}")
        mask, maskcnt = extractor.create_color_mask(blue_image, mask_name=color, hsv_ranges=ranges,
                                                         bbox=target_bbox)
        if maskcnt < 10:  # Ignore small masks
            continue
        buffered_mask = extractor.create_buffered_mask(mask_name=color, buffer_size=buffer_size)

        decomposer = AdaptiveRegionDecomposition(
            image, buffered_mask, min_size=min_size
        )
        decomposer.run()
