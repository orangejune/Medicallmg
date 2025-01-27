import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev
# import pims
# import tifffile
import SimpleITK as sitk

class ContourExtraction:
    def __init__(self, full_image, blue_image):
        self.full_image = full_image
        self.blue_image = blue_image

    def extract_boundaries_and_filter_by_mask(self, mask, kernel_size=(5, 5), morph_kernel=(5, 5), min_area=100, smooth=False,  min_threshold=None, max_threshold=None):
        """
        Extract contours for the entire image and:
        - Store the valid portion of each contour within the buffer zone.
        - Store the entire contour if at least one point lies within the buffer zone.

        Args:
            mask (numpy.ndarray): Binary mask defining the region of interest.
            morph_kernel (tuple): Kernel size for morphological operations to clean the mask.
            min_area (int): Minimum area to keep a contour. Smaller areas are ignored.
            smooth (bool): Whether to smooth the contours.

        Returns:
            tuple: Two lists:
                - valid_parts_contours: Portions of contours within the buffer zone.
                - full_contours: Entire contours that intersect with the buffer zone.
        """
        # Step 1: Convert full image to grayscale if needed
        if len(self.full_image.shape) == 3:
            gray_image = cv2.cvtColor(self.full_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.full_image

        # Step 2: Threshold the image to create a binary mask
        # if min_threshold is not None and max_threshold is not None:
        #     binary_image = cv2.inRange(gray_image, min_threshold, max_threshold)
        # else:
        _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Step 3: Find contours for the entire image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Step 4: Apply morphological operations to clean the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


        # Step 5: Initialize lists for valid portions and full contours
        valid_parts_contours = []
        full_contours = []

        # Step 6: Process each contour
        for contour in contours:
            # Check if any point in the contour lies within the mask
            mask_points = []
            for point in contour:
                px, py = point[0]
                if cleaned_mask[py, px] > 0:  # Keep only points within the mask
                    mask_points.append([px, py])

            # Store the valid portion within the buffer zone
            if len(mask_points) > 1:  # Ensure the portion has enough points
                valid_part = np.array(mask_points, dtype=np.int32).reshape(-1, 1, 2)
                if cv2.contourArea(valid_part) >= min_area:  # Filter by area
                    valid_parts_contours.append(valid_part)

            # Store the entire contour if any point is in the buffer zone
            if any(cleaned_mask[py, px] > 0 for px, py in contour[:, 0]) and cv2.contourArea(contour) >= min_area:
                full_contours.append(contour)

        # Optional: Smooth the contours
        if smooth:
            valid_parts_contours = [self.smooth_contour(cnt) for cnt in valid_parts_contours]
            full_contours = [self.smooth_contour(cnt) for cnt in full_contours]

        print(f"Extracted {len(valid_parts_contours)} valid parts and {len(full_contours)} full contours.")
        return valid_parts_contours, full_contours

    ##todo: reorder points to avoid a line added after filter  -- not  tested ...
    def reorder_contour_points(self, contour, mask):
        """
        Reorder contour points to maintain natural continuity, avoiding artificial lines, and ensuring a single polyline.

        Args:
            contour (numpy.ndarray): Original contour points (Nx1x2).
            mask (numpy.ndarray): Binary mask defining the buffer zone.

        Returns:
            numpy.ndarray: A single reordered contour.
        """
        # Step 1: Identify valid segments
        valid_points = []
        current_segment = []

        for i, point in enumerate(contour):
            px, py = point[0]
            if mask[py, px] > 0:  # Point is within the mask
                current_segment.append([px, py])
            else:
                if current_segment:
                    valid_points.append(current_segment)
                    current_segment = []

        # Add the last segment if it exists
        if current_segment:
            valid_points.append(current_segment)

        # Step 2: Reorder points to form a single polyline --todo: this part is not quite right
        if len(valid_points) > 1:
            # Combine the last segment and the first segment for continuity
            reordered_points = valid_points[-1] + valid_points[0]
            # Add the middle segments (if any)
            for segment in valid_points[1:-1]:
                reordered_points.extend(segment)
        else:
            # If only one segment exists, no reordering is needed
            reordered_points = valid_points[0]

        # Step 3: Return the reordered points as a single contour
        return np.array(reordered_points, dtype=np.int32).reshape(-1, 1, 2)

    def VizContours(self, contours, annotated_image):
        # Visualize and annotate contours with labels and directions
        for i, line in enumerate(contours):
            # Draw the contour as a polyline
            cv2.polylines(annotated_image, [line], isClosed=False, color=(255, 255, 0), thickness=1)

            # Label the contour with its index
            if len(line) > 0:
                # Use the first point of the contour for the label
                label_position = tuple(line[0][0])
                cv2.putText(annotated_image, f"#{i}", label_position, cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.25, color=(0, 255, 0), thickness=1)

                # Draw arrows to indicate the direction
                for j in range(0, len(line) - 1, max(1, len(line) // 20)):  # Add arrows at regular intervals
                    start_point = tuple(line[j][0])
                    end_point = tuple(line[j + 1][0])
                    cv2.arrowedLine(annotated_image, start_point, end_point, color=(0, 0, 255), thickness=1,
                                    tipLength=0.3)


    ##todo:Not tested  a function to perform extraction per region (w,d determine the number of zones try to divide images vertically)
    def extract_contours_by_zones(self, mask, n_zones, thresholds=None, min_area=100):
        """
        Divide the image vertically into zones, apply contour extraction for each zone with adaptive or specified thresholds, and merge results.

        Args:
            mask (numpy.ndarray): Binary mask defining the region of interest.
            n_zones (int): Number of vertical zones to divide the image into.
            thresholds (list of tuple or None): List of (min_threshold, max_threshold) for each zone. If None, Otsu's method is applied.
            min_area (int): Minimum area to keep a contour.

        Returns:
            list: Merged contours from all zones.
        """
        height, width = self.full_image.shape[:2]
        zone_width = width // n_zones
        merged_contours = []

        for i in range(n_zones):
            # Define the vertical range for the current zone
            x_start = i * zone_width
            x_end = (i + 1) * zone_width if i < n_zones - 1 else width

            # Extract the zone from the full image and mask
            zone_image = self.full_image[:, x_start:x_end]
            zone_mask = mask[:, x_start:x_end]

            # Apply thresholding
            if thresholds is not None:
                # Use provided thresholds for the current zone
                min_threshold, max_threshold = thresholds[i]
                binary_image = cv2.inRange(zone_image, min_threshold, max_threshold)
            else:
                # Use Otsu's method for adaptive thresholding
                _, binary_image = cv2.threshold(zone_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # Find contours in the current zone
            contours, _ = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours based on area and adjust coordinates
            for contour in contours:
                if cv2.contourArea(contour) >= min_area:
                    contour[:, 0, 0] += x_start  # Adjust x-coordinates to global scale
                    merged_contours.append(contour)

        print(f"Extracted {len(merged_contours)} contours from {n_zones} zones.")
        return merged_contours



