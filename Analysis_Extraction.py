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
         

    def extract_boundaries_and_filter_by_mask(self, mask, kernel_size=(5, 5), morph_kernel=(5, 5), min_area=100, smooth=False, min_threshold=None, max_threshold=None):
        """
        Extract contours for the entire image and:
        - Store the valid portion of each contour within the buffer zone.
        - Store the entire contour if at least one point lies within the buffer zone.
        - Reorder the points using `reorder_contour_points`.
        - Return multiple segments if gaps exist.
        - Skip inner contours (contours inside other contours).

        Args:
            mask (numpy.ndarray): Binary mask defining the region of interest.
            kernel_size (tuple): Kernel size for image preprocessing.
            morph_kernel (tuple): Kernel size for morphological operations.
            min_area (int): Minimum area to keep a contour. Smaller areas are ignored.
            smooth (bool): Whether to smooth the contours.
            min_threshold (int or None): Lower bound for thresholding.
            max_threshold (int or None): Upper bound for thresholding.

        Returns:
            tuple:
                - valid_parts_contours (list of numpy.ndarray): Reordered valid contours.
                - full_contours (list of numpy.ndarray): Entire contours that intersect with the buffer zone.
                - segment_contours (list of lists of numpy.ndarray): List of contour segments for verification.
        """
        # Step 1: Convert full image to grayscale if needed
        if len(self.full_image.shape) == 3:
            gray_image = cv2.cvtColor(self.full_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.full_image

        # Step 2: Apply thresholding
        if min_threshold is not None and max_threshold is not None:
            binary_image = cv2.inRange(gray_image, min_threshold, max_threshold)
        else:
            _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Step 3: Find contours **with hierarchy** (to remove inner contours)
        contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Step 4: Apply morphological operations to clean the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Step 5: Initialize lists for valid portions and full contours
        valid_parts_contours = []
        full_contours = []
        segment_contours = []

        # Step 6: Process each contour (keep only outer contours)
        for i, contour in enumerate(contours):
            # Skip inner contours (contours inside another contour)
            if hierarchy[0][i][3] != -1:
                continue  # This contour is inside another, so we skip it

            # Reorder the contour and find segments
            reordered, segments = self.reorder_contour_points(contour, cleaned_mask)

            # If reordering worked and the area is valid, add it to the valid contours list
            if reordered is not None and cv2.contourArea(reordered) >= min_area:
                valid_parts_contours.append(reordered)
                segment_contours.append(segments)

            # Store the entire contour if any part is in the buffer
            if any(cleaned_mask[py, px] > 0 for px, py in contour[:, 0]) and cv2.contourArea(contour) >= min_area:
                full_contours.append(contour)

        # Step 7: Optional smoothing
        if smooth:
            valid_parts_contours = [self.smooth_contour(cnt) for cnt in valid_parts_contours]
            full_contours = [self.smooth_contour(cnt) for cnt in full_contours]

        print(f"Extracted {len(valid_parts_contours)} reordered contours and {len(segment_contours)} segmented contours from {len(full_contours)} full outer contours.")
        return valid_parts_contours, full_contours, segment_contours




    def reorder_contour_points(self, contour, mask):
        """
        Reorder contour points to maintain natural continuity, avoiding artificial lines.
        Supports multiple gaps and returns both a single reordered polyline and separate segments.

        Args:
            contour (numpy.ndarray): Original contour points (Nx1x2).
            mask (numpy.ndarray): Binary mask defining the buffer zone.

        Returns:
            tuple:
                - reordered_contour (numpy.ndarray): A single reordered contour following continuity.
                - contour_segments (list of numpy.ndarray): Separate contour segments split at gaps.
        """
        # Step 1: Create a dictionary to track point indices and coordinates
        point_dict = {i: tuple(contour[i][0]) for i in range(len(contour))}

        # Step 2: Identify valid points within the mask
        valid_points = {i: point_dict[i] for i in point_dict if mask[point_dict[i][1], point_dict[i][0]] > 0}
        valid_indices = sorted(valid_points.keys())

        if not valid_indices:
            return None, []  # No valid points remain

        # Step 3: Detect all gaps in valid indices
        gap_indices = []
        for i in range(1, len(valid_indices)):
            if valid_indices[i] - valid_indices[i - 1] > 1:  # Found a gap
                gap_indices.append(i)

        # Step 4: Split valid points into segments
        contour_segments = []
        start_idx = 0
        for gap_idx in gap_indices:
            segment_indices = valid_indices[start_idx:gap_idx]
            segment = [valid_points[i] for i in segment_indices]
            if len(segment) > 1:
                contour_segments.append(np.array(segment, dtype=np.int32).reshape(-1, 1, 2))
            start_idx = gap_idx

        # Add the final segment
        segment_indices = valid_indices[start_idx:]
        segment = [valid_points[i] for i in segment_indices]
        if len(segment) > 1:
            contour_segments.append(np.array(segment, dtype=np.int32).reshape(-1, 1, 2))

        # Step 5: Reorder points by choosing the segment after the largest gap as the new starting point
        if len(contour_segments) > 1:
            reordered_contour = np.vstack(contour_segments[::-1])  # Reverse segment order for natural flow
        else:
            reordered_contour = contour_segments[0] if contour_segments else None

        return reordered_contour, contour_segments


    def visualize_contours(self, full_contours, valid_parts_contours, segment_contours, valid_parts=None):
        """
        Visualize contours using OpenCV (cv2.imshow) instead of Matplotlib.

        Args:
            full_contours (list of numpy.ndarray): Full contours intersecting the buffer zone.
            valid_parts_contours (list of numpy.ndarray): Reordered valid contours.
            segment_contours (list of lists of numpy.ndarray): List of segmented contours.
            valid_parts (list of numpy.ndarray or None): Additional valid parts to visualize (default: None).

        Returns:
            None: Displays the image with contours using OpenCV.
        """
        # Ensure full_image is in BGR format for visualization
        if len(self.full_image.shape) == 2:
            annotated_image = cv2.cvtColor(self.full_image, cv2.COLOR_GRAY2BGR)
        else:
            annotated_image = self.full_image.copy()
        print()

        # Draw full contours in Yellow
        for line in full_contours:
            cv2.polylines(annotated_image, [line], isClosed=False, color=(255, 255, 0), thickness=4)

        # Draw reordered contours in Green
        for line in valid_parts_contours:
            cv2.polylines(annotated_image, [line], isClosed=False, color=(0, 255, 0), thickness=2)

        # Draw segmented contours in Red
        # for segments in segment_contours:
        #     for line in segments:
        #         cv2.polylines(annotated_image, [line], isClosed=False, color=(0, 0, 255), thickness=2)

        # Draw valid parts in Cyan (if provided)
        if valid_parts is not None:
            for line in valid_parts:
                cv2.polylines(annotated_image, [line], isClosed=False, color=(255, 0, 255), thickness=2)  # Cyan

        # Display the image using OpenCV
        cv2.imshow("Contours Visualization", annotated_image)
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        cv2.destroyAllWindows()  # Close the window after key press



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



