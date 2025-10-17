import cv2
import numpy as np

class MaskBufferExtractor:
    def __init__(self, full_image, blue_image):
        """
        Initialize the MaskBufferExtractor.

        Args:
            full_image (numpy.ndarray): Full grayscale image.
            blue_image (numpy.ndarray): Image containing labeled regions.
        """
        self.full_image = full_image
        self.blue_image = blue_image
        self.masks = {}
        self.buffers = {}

    def create_color_mask(self, image, mask_name, hsv_ranges, bbox=None):
        """
        Create a binary mask to detect regions of a specific color using HSV ranges, restricted to a bounding box.

        Args:
            image (numpy.ndarray): Input BGR image.
            mask_name (str): Unique name for the mask.
            hsv_ranges (list): List of HSV range tuples [(lower, upper), ...].
            bbox (tuple or None): Bounding box as (left, top, right, bottom). If None, no restriction is applied.

        Returns:
            numpy.ndarray: Binary mask.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

        # Apply each HSV range and combine the results
        for lower, upper in hsv_ranges:
            mask = cv2.inRange(hsv_image, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Restrict the mask to the bounding box if specified
        if bbox is not None:
            left, top, right, bottom = bbox
            roi = np.zeros_like(combined_mask)
            roi[top:bottom, left:right] = combined_mask[top:bottom, left:right]
            combined_mask = roi

        self.masks[mask_name] = combined_mask
        print(f"Mask '{mask_name}' created within bounding box {bbox} with {cv2.countNonZero(combined_mask)} non-zero pixels.")
        cnt =  cv2.countNonZero(combined_mask)
        return combined_mask, cnt

    def create_buffered_mask(self, mask_name, buffer_size):
        """
        Create a buffered mask by dilating the specified mask.

        Args:
            mask_name (str): Name of the existing mask.
            buffer_size (int): Buffer size in pixels.

        Returns:
            numpy.ndarray: Buffered mask.
        """
        if mask_name not in self.masks:
            raise ValueError(f"Mask '{mask_name}' not found.")

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (buffer_size, buffer_size))
        buffered_mask = cv2.dilate(self.masks[mask_name], kernel, iterations=1)
        self.buffers[mask_name] = buffered_mask
        print(f"Buffered mask for '{mask_name}' created.")
        return buffered_mask


