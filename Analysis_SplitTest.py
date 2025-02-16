import cv2
import numpy as np
import matplotlib.pyplot as plt

import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageContourExtractor:
    def __init__(self, image_path, grid_size=(4, 4)):
        """
        Initialize the ImageContourExtractor with an image and grid size.

        Parameters:
            image_path (str): Path to the input image.
            grid_size (tuple): Tuple (W, H) specifying the number of horizontal and vertical divisions.
        """
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.grid_size = grid_size
        self.h, self.w = self.image.shape
        self.W, self.H = grid_size
        self.h_step, self.w_step = self.h // self.H, self.w // self.W
        self.sub_images = []
        self.processed_regions = []

    def divide_image(self):
        """Divide the image into a grid of sub-regions."""
        for i in range(self.H):
            for j in range(self.W):
                sub_img = self.image[i * self.h_step:(i + 1) * self.h_step,
                          j * self.w_step:(j + 1) * self.w_step]
                self.sub_images.append((sub_img, (i, j)))

    def process_region(self, region):
        """
        Apply Otsu's thresholding and extract contours from a region.

        Parameters:
            region (numpy.ndarray): Image region.

        Returns:
            list: List of detected contours.
        """
        _, thresh = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def extract_contours(self):
        """Extract contours from all image sub-regions."""
        self.processed_regions = [(self.process_region(sub_img), pos) for sub_img, pos in self.sub_images]

    def draw_grid_boundaries(self, output_image):
        """Draw the grid boundaries on the output image."""
        for i in range(1, self.H):  # Draw horizontal lines
            y = i * self.h_step
            cv2.line(output_image, (0, y), (self.w, y), (255, 0, 0), 1)  # Blue lines

        for j in range(1, self.W):  # Draw vertical lines
            x = j * self.w_step
            cv2.line(output_image, (x, 0), (x, self.h), (255, 0, 0), 1)  # Blue lines

    def visualize_results(self):
        """Visualize the extracted contours and region boundaries on the original image."""
        output_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

        # Draw contours
        for contours, (i, j) in self.processed_regions:
            for contour in contours:
                contour[:, 0, 0] += j * self.w_step  # Adjust x-coordinates
                contour[:, 0, 1] += i * self.h_step  # Adjust y-coordinates
                cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 3)  # Green contours

        # Draw grid boundaries
        self.draw_grid_boundaries(output_image)

        # Show final result
        plt.figure(figsize=(10, 6))
        plt.imshow(output_image)
        plt.title("Extracted Contours and Region Boundaries")
        plt.axis("off")
        plt.show()

    def run(self):
        """Run the full pipeline: divide, extract contours, and visualize."""
        self.divide_image()
        self.extract_contours()
        self.visualize_results()




# Usage Example
image_path = r"KD/New/KD-3-dengjinyi/KD-3-dengjinyi/Series-001/jpg-img-00001-00003.jpg"  # Replace with your image path
extractor = ImageContourExtractor(image_path, grid_size=(4, 4))
extractor.run()
