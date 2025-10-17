import cv2
import numpy as np

class DistanceChecker:
    def __init__(self, restricted_contours, full_image):
        self.restricted_contours = restricted_contours
        self.full_image = full_image

    def sample_contour_points(self, contour, num_points=200):
        """
        Increase sampling of contour points using linear interpolation.

        Args:
            contour: The original contour points.
            num_points: The number of sampled points.

        Returns:
            sampled_contour: The resampled contour points.
        """
        contour = contour.reshape(-1, 2)
        distance = np.cumsum(np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1)))
        distance = np.insert(distance, 0, 0)  # Include starting point
        interpolated_points = np.linspace(0, distance[-1], num_points)
        sampled_contour = np.vstack([
            np.interp(interpolated_points, distance, contour[:, 0]),
            np.interp(interpolated_points, distance, contour[:, 1])
        ]).T
        return sampled_contour.astype(np.int32).reshape(-1, 1, 2)

    def debug_distances(self, num_points=200):
        """
        Compute distances by sampling contours and finding the maximum diameter.

        Args:
            num_points: Number of sampled points for each contour.

        Returns:
            distances: List of distances from sampled_contour1 to sampled_contour2.
            nearest_points: List of nearest points on sampled_contour2.
            max_diameter: Maximum distance (diameter).
            max_points: The pair of points defining the maximum diameter.
            sampled_contour1: The sampled points of contour1.
            sampled_contour2: The sampled points of contour2.
        """
        if len(self.restricted_contours) < 2:
            raise ValueError("Less than two restricted contours detected.")

        # Assign contour1 as the shorter one based on arc length
        if cv2.arcLength(self.restricted_contours[0], closed=False) < cv2.arcLength(self.restricted_contours[1],
                                                                                    closed=False):
            contour1, contour2 = self.restricted_contours[0], self.restricted_contours[1]
        else:
            contour1, contour2 = self.restricted_contours[1], self.restricted_contours[0]

        # Resample contours for uniform point density
        sampled_contour1 = self.sample_contour_points(contour1, num_points=num_points)
        sampled_contour2 = self.sample_contour_points(contour2, num_points=num_points)

        max_diameter = 0
        max_points = (None, None)
        distances = []
        nearest_points = []

        for point1 in sampled_contour1.reshape(-1, 2):
            # Find the closest point on contour2 to point1
            contour2_points = sampled_contour2.reshape(-1, 2)
            diff = contour2_points - point1
            squared_distances = np.sum(diff ** 2, axis=1)
            min_idx = np.argmin(squared_distances)
            closest_point = contour2_points[min_idx]
            distance = np.sqrt(squared_distances[min_idx])

            distances.append(distance)
            nearest_points.append(closest_point)

            # Update max diameter if necessary
            if distance > max_diameter:
                max_diameter = distance
                max_points = (point1, closest_point)

        return distances, nearest_points, max_diameter, max_points, sampled_contour1, sampled_contour2

    def visualize_debug_distances(self, sampled_contour1, sampled_contour2, distances, nearest_points,
                                  max_diameter=None, max_points=None, save_path=None):
        """
        Visualize distances from contour1 to contour2, highlighting the maximum diameter and labeling distances.

        Args:
            sampled_contour1: The sampled points of contour1.
            sampled_contour2: The sampled points of contour2.
            distances: List of distances from contour1 to contour2.
            nearest_points: List of nearest points on contour2.
            max_diameter: Maximum inner diameter (optional).
            max_points: The pair of points defining the maximum diameter (optional).
            save_path: Path to save the visualization.
        """
        # Ensure the image is grayscale before converting to BGR
        if len(self.full_image.shape) == 2:  # Grayscale image
            annotated_image = cv2.cvtColor(self.full_image, cv2.COLOR_GRAY2BGR)
        else:  # Already in BGR format
            annotated_image = self.full_image.copy()

        # Draw contours
        cv2.polylines(annotated_image, [sampled_contour1], isClosed=False, color=(0, 255, 0), thickness=2)  # Green
        cv2.polylines(annotated_image, [sampled_contour2], isClosed=False, color=(255, 0, 0), thickness=2)  # Red

        # Draw and label distances
        for i, (distance, point, nearest_point) in enumerate(
                zip(distances, sampled_contour1.reshape(-1, 2), nearest_points)):
            if distance == float('inf') or nearest_point is None:
                continue  # Skip invalid distances

            p1 = tuple(map(int, point))
            p2 = tuple(map(int, nearest_point))
            cv2.line(annotated_image, p1, p2, (0, 0, 255), thickness=1)  # Blue line for distance
            # midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)  # Midpoint of the line
            # cv2.putText(annotated_image, f"{distance:.2f}", midpoint, cv2.FONT_HERSHEY_SIMPLEX,
            #             0.4, (255, 255, 255), 1, cv2.LINE_AA)  # Annotate with distance value

        # Highlight the maximum distance
        if max_points and max_points[0] is not None and max_points[1] is not None:
            p1 = tuple(map(int, max_points[0]))
            p2 = tuple(map(int, max_points[1]))
            cv2.line(annotated_image, p1, p2, (0, 255, 255), thickness=2)  # Yellow line for max distance
            cv2.circle(annotated_image, p1, 5, (0, 0, 255), -1)  # Red circle for max point on contour1
            cv2.circle(annotated_image, p2, 5, (255, 255, 0), -1)  # Yellow circle for max point on contour2
            midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(annotated_image, f"Max: {max_diameter:.2f}", midpoint, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)  # Annotate with max diameter value

        # Display and optionally save the visualization
        cv2.imshow("Distances and Maximum Diameter", annotated_image)
        if save_path:
            cv2.imwrite(save_path, annotated_image)
            print(f"Visualization saved at {save_path}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    ##todo: create a function to smooth contours and identify line-line distance using large smoothy lines



