import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


class MidlineExtractor:
    def __init__(self, n_points=100, smoothing=True, smoothing_window=11, smoothing_polyorder=2):
        """
        Initializes the MidlineExtractor class.

        Parameters:
        - n_points (int): Number of points to resample contours to ensure correspondence.
        - smoothing (bool): Whether to apply smoothing to the computed midline.
        - smoothing_window (int): Window length for smoothing (must be odd).
        - smoothing_polyorder (int): Polynomial order for smoothing.
        """
        self.n_points = n_points
        self.smoothing = smoothing
        self.smoothing_window = smoothing_window
        self.smoothing_polyorder = smoothing_polyorder

    def resample_contour_by_length(contour, n_points):
        """
        Resamples a contour based on its cumulative arc length.

        Parameters:
        - contour (np.ndarray): Original contour points (Nx2).
        - n_points (int): Number of points to resample to.

        Returns:
        - np.ndarray: Resampled contour points (n_points x 2).
        """
        # Compute cumulative arc length
        distances = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
        cumulative_length = np.insert(np.cumsum(distances), 0, 0)

        # Normalize cumulative lengths to [0, 1]
        total_length = cumulative_length[-1]
        normalized_length = cumulative_length / total_length

        # Define evenly spaced points in the normalized length
        target_length = np.linspace(0, 1, n_points)

        # Interpolate x and y coordinates at target lengths
        interp_x = np.interp(target_length, normalized_length, contour[:, 0])
        interp_y = np.interp(target_length, normalized_length, contour[:, 1])

        return np.vstack((interp_x, interp_y)).T

    @staticmethod
    def resample_contour(self, contour, n_points):
        """
        Resamples a contour to have a fixed number of points based on arc length.

        Parameters:
        - contour (np.ndarray): Original contour points (Nx2).
        - n_points (int): Number of points to resample to.

        Returns:
        - np.ndarray: Resampled contour points (n_points x 2).
        """
        return self.resample_contour_by_length(contour, n_points)

    def align_contour_directions(self, contour1, contour2):
        """
        Aligns the directions of two contours to ensure they are consistent.

        Parameters:
        - contour1 (np.ndarray): First contour points (Nx2).
        - contour2 (np.ndarray): Second contour points (Nx2).

        Returns:
        - np.ndarray, np.ndarray: Aligned contours.
        """

        def contour_direction(contour):
            area = 0.5 * np.sum(contour[:-1, 0] * contour[1:, 1] - contour[1:, 0] * contour[:-1, 1])
            return 1 if area > 0 else -1

        # Check and align directions
        if contour_direction(contour1) != contour_direction(contour2):
            contour2 = contour2[::-1]  # Reverse contour2 if directions differ

        return contour1, contour2

    def compute_midline(self, contour1, contour2):
        """
        Computes the midline between two contours.

        Parameters:
        - contour1 (np.ndarray): First contour points (Nx2).
        - contour2 (np.ndarray): Second contour points (Nx2).

        Returns:
        - np.ndarray: Midline points (Nx2).
        """
        # Resample contours to have the same number of points
        resampled_contour1 = self.resample_contour(contour1, self.n_points)
        resampled_contour2 = self.resample_contour(contour2, self.n_points)

        # Compute the midline as the average of the corresponding points
        midline = (resampled_contour1 + resampled_contour2) / 2

        # Apply optional smoothing
        if self.smoothing:
            midline[:, 0] = savgol_filter(midline[:, 0], self.smoothing_window, self.smoothing_polyorder)
            midline[:, 1] = savgol_filter(midline[:, 1], self.smoothing_window, self.smoothing_polyorder)

        return midline



    @staticmethod
    def perpendicular_vector(v):
        """Computes a vector perpendicular to the given vector."""
        return np.array([-v[1], v[0]])

    def compute_distances(self, midline, contour1, contour2, step=10):
        """
        Computes distances from perpendicular lines at midline points to the contour walls.

        Parameters:
        - midline (np.ndarray): Midline points (Nx2).
        - contour1 (np.ndarray): First contour points (Nx2).
        - contour2 (np.ndarray): Second contour points (Nx2).
        - step (int): Step size to sample midline points for perpendicular line computation.

        Returns:
        - list: Distances for each sampled perpendicular line.
        - float: Maximum distance.
        """
        distances = []
        for i in range(0, len(midline), step):
            # Perpendicular line calculation
            p = midline[i]
            if i < len(midline) - 1:
                direction = midline[i + 1] - p
            else:
                direction = midline[i] - midline[i - 1]
            perp_vector = self.perpendicular_vector(direction / np.linalg.norm(direction))
            perp_line_start = p - 100 * perp_vector  # Extend perpendicular line
            perp_line_end = p + 100 * perp_vector

            # Find intersection with contour1 and contour2
            def find_intersection(line, contour):
                min_dist = float('inf')
                intersection = None
                for j in range(len(contour) - 1):
                    seg_start = contour[j]
                    seg_end = contour[j + 1]
                    denom = (line[1][0] - line[0][0]) * (seg_end[1] - seg_start[1]) - \
                            (line[1][1] - line[0][1]) * (seg_end[0] - seg_start[0])
                    if denom == 0:
                        continue  # Parallel lines
                    ua = ((seg_end[0] - seg_start[0]) * (line[0][1] - seg_start[1]) - \
                          (seg_end[1] - seg_start[1]) * (line[0][0] - seg_start[0])) / denom
                    if 0 <= ua <= 1:
                        px = line[0][0] + ua * (line[1][0] - line[0][0])
                        py = line[0][1] + ua * (line[1][1] - line[0][1])
                        dist = np.linalg.norm(np.array([px, py]) - p)
                        if dist < min_dist:
                            min_dist = dist
                            intersection = np.array([px, py])
                return intersection, min_dist

            intersection1, dist1 = find_intersection((perp_line_start, perp_line_end), contour1)
            intersection2, dist2 = find_intersection((perp_line_start, perp_line_end), contour2)

            if intersection1 is not None and intersection2 is not None:
                distances.append(dist1 + dist2)

        max_distance = max(distances) if distances else 0
        return distances, max_distance



    def compute_midline(self, contour1, contour2):
        """
        Computes the midline between two contours.

        Parameters:
        - contour1 (np.ndarray): First contour points (Nx2).
        - contour2 (np.ndarray): Second contour points (Nx2).

        Returns:
        - np.ndarray: Midline points (Nx2).
        """
        # Align contour directions
        contour1, contour2 = self.align_contour_directions(contour1, contour2)

        # Resample contours to have the same number of points
        resampled_contour1 = self.resample_contour(contour1, self.n_points)
        resampled_contour2 = self.resample_contour(contour2, self.n_points)

        # Compute the midline as the average of the corresponding points
        midline = (resampled_contour1 + resampled_contour2) / 2

        # Apply optional smoothing
        if self.smoothing:
            midline[:, 0] = savgol_filter(midline[:, 0], self.smoothing_window, self.smoothing_polyorder)
            midline[:, 1] = savgol_filter(midline[:, 1], self.smoothing_window, self.smoothing_polyorder)

        return midline

    def extract_midline_and_distances(self, contour1, contour2, step=10):
        """
        Computes the midline and the distances from the midline to the contours.

        Parameters:
        - contour1 (np.ndarray): First contour points (Nx2).
        - contour2 (np.ndarray): Second contour points (Nx2).
        - step (int): Step size to sample midline points.

        Returns:
        - np.ndarray: Midline points (Nx2).
        - list: Distances for each sampled perpendicular line.
        - float: Maximum distance.
        """
        midline = self.compute_midline(contour1, contour2)
        distances, max_distance = self.compute_distances(midline, contour1, contour2, step)
        return midline, distances, max_distance




# Example usage
##process: a pair of contours -- sort contour directions -- sample contours by length -- compute middle line  -- compute distance -- find the max distance
##todo: only compute at smooth middle locacations ---e.g., xx pts from the start/end locs?
if __name__ == "__main__":
    # Example contours
    contour1 = np.array([[10, 10], [20, 15], [30, 20]])
    contour2 = np.array([[12, 12], [22, 18], [32, 22]])

    # Initialize the midline extractor
    extractor = MidlineExtractor(n_points=50, smoothing=True)

    # Compute the midline and distances
    midline, distances, max_distance = extractor.extract_midline_and_distances(contour1, contour2, step=5)

    # Display the result
    print("Midline points:\n", midline)
    print("Distances:\n", distances)
    print("Maximum distance:\n", max_distance)
