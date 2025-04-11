import cv2
import numpy as np
import SimpleITK as sitk
import time

class ContourTrimmer:
    def __init__(self, image, contour):
        self.image = image
        self.points = []  # Store the start and end points of each line
        self.lines = []  # Store all drawn lines
        self.current_line = []  # Points of the current line being drawn
        self.contour = contour  # Original contour

    def trim_contour(self):
        """
        Trim the contour by extracting the part between two intersecting lines.

        Returns:
            list: A list of trimmed contours.
        """
        lines = self.get_lines()
        result_contours = []
        for contour in self.contour:
            # Extract the contour between the intersections of the two lines
            result_contour = self.extract_contour_between_intersections(contour, lines[0], lines[1])
            if result_contour is not None:
                # Draw the result contour on the image
                cv2.polylines(self.image, [result_contour], isClosed=False, color=(0, 255, 255), thickness=2)

                # Mark the intersections
                intersection1 = self.find_contour_intersection(contour, lines[0])
                intersection2 = self.find_contour_intersection(contour, lines[1])
                if intersection1:
                    cv2.circle(self.image, (int(intersection1[0][0]), int(intersection1[0][1])), 3, (255, 255, 0), -1)
                if intersection2:
                    cv2.circle(self.image, (int(intersection2[0][0]), int(intersection2[0][1])), 3, (255, 255, 0), -1)

                result_contours.append(result_contour)
        # Display the result
        cv2.imshow('Result', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return result_contours

    def get_lines(self):
        """
        Get two lines by clicking on the image.

        Returns:
            list: A list of two lines, each represented by two points.
        """
        def mouse_callback(event, x, y, flags, param):
            # Define the mouse callback function
            if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
                if len(self.current_line) < 2:  # Each line needs two points
                    self.current_line.append((x, y))  # Record the current point of the line
                    # print(f"Click coordinates: ({x}, {y})")

                    if len(self.current_line) == 2:  # If the current line has two points
                        self.lines.append(self.current_line.copy())  # Add the current line to all lines
                        self.current_line.clear()  # Clear the current line, prepare to draw the next line

                        if len(self.lines) == 2:  # If two lines have been drawn
                            cv2.destroyAllWindows()

        # Create a window and bind the mouse callback function
        cv2.namedWindow('Please click to draw the line:')
        cv2.setMouseCallback('Please click to draw the line:', mouse_callback)
        # Display the image and draw lines in real-time
        while True:
            temp_image = self.image.copy()  # Use a copy of the original image each loop
            for contour in self.contour:
                cv2.polylines(temp_image, [contour], isClosed=False, color=(0, 255, 0), thickness=2)

            # Draw all completed lines
            for line in self.lines:
                cv2.line(temp_image, line[0], line[1], (0, 0, 255), 2)

            # Draw the current line being drawn (if any points)
            if len(self.current_line) == 1:
                cv2.line(temp_image, self.current_line[0], self.current_line[0], (0, 0, 255), 2)  # Draw a point
            elif len(self.current_line) == 2:
                cv2.line(temp_image, self.current_line[0], self.current_line[1], (0, 0, 255), 2)  # Draw a line

            # Display the image
            cv2.imshow('Please click to draw the line:', temp_image)

            # Detect exit conditions
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Press ESC key
                break
            if len(self.lines) == 2:  # Two lines have been drawn
                for line in self.lines:
                    cv2.line(temp_image, line[0], line[1], (0, 0, 255), 2)
                # cv2.imshow('Image Window', temp_image)
                # cv2.waitKey(1000)
                break

        cv2.destroyAllWindows()
        return self.lines

    def line_intersection(self, line1, line2):
        """
        Calculate the intersection point of two line segments.

        Args:
            line1: ((x1, y1), (x2, y2))
            line2: ((x3, y3), (x4, y4))

        Returns:
            tuple: Intersection point (x, y) or None if no intersection.
        """
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2

        # Calculate the denominator
        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

        # If the denominator is 0, the lines are parallel or collinear
        if denom == 0:
            return None

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

        # Check if the intersection point is on both line segments
        if 0 <= ua <= 1 and 0 <= ub <= 1:
            x = x1 + ua * (x2 - x1)
            y = y1 + ua * (y2 - y1)
            return (x, y)
        return None

    def find_contour_intersection(self, contour, line):
        """
        Find the intersection point of a contour with a line segment.

        Args:
            contour: Contour points, shape (n, 1, 2)
            line: ((x1, y1), (x2, y2))

        Returns:
            tuple: Intersection point (coordinates, index of the previous contour point) or None.
        """
        contour_points = contour.reshape(-1, 2)

        for i in range(len(contour_points) - 1):
            pt1 = contour_points[i]
            pt2 = contour_points[i + 1]
            contour_segment = (tuple(pt1), tuple(pt2))

            intersection = self.line_intersection(contour_segment, line)
            if intersection is not None:
                return intersection, i

        # Check the last segment of a closed contour
        if len(contour_points) > 2:
            pt1 = contour_points[-1]
            pt2 = contour_points[0]
            contour_segment = (tuple(pt1), tuple(pt2))

            intersection = self.line_intersection(contour_segment, line)
            if intersection is not None:
                return intersection, len(contour_points) - 1

        return None

    def extract_contour_between_intersections(self, contour, line1, line2):
        """
        Extract the part of the contour between two intersection points.

        Args:
            contour: Contour points, shape (n, 1, 2)
            line1: ((x1, y1), (x2, y2))
            line2: ((x3, y3), (x4, y4))

        Returns:
            numpy.ndarray: The extracted contour points.
        """
        # Find the two intersection points
        intersection1 = self.find_contour_intersection(contour, line1)
        intersection2 = self.find_contour_intersection(contour, line2)

        if intersection1 is None or intersection2 is None:
            return None

        (point1, idx1), (point2, idx2) = intersection1, intersection2

        # Convert the contour to a list of points
        contour_points = contour.reshape(-1, 2).tolist()

        # Determine the order of the intersection points
        if idx1 > idx2 or (idx1 == idx2 and
                           np.linalg.norm(np.array(contour_points[idx1]) - np.array(point1)) >
                           np.linalg.norm(np.array(contour_points[idx1]) - np.array(point2))):
            point1, idx1, point2, idx2 = point2, idx2, point1, idx1

        # Build the new contour
        new_contour = []

        # Add the first intersection point
        new_contour.append(point1)

        # Add the intermediate points
        if idx1 < idx2:
            # Normal order
            for i in range(idx1 + 1, idx2 + 1):
                new_contour.append(contour_points[i % len(contour_points)])
        else:
            # Need to wrap around
            for i in range(idx1 + 1, len(contour_points)):
                new_contour.append(contour_points[i])
            for i in range(0, idx2 + 1):
                new_contour.append(contour_points[i])

        # Add the second intersection point
        new_contour.append(point2)

        # Convert to OpenCV contour format
        result = np.array(new_contour).reshape(-1, 1, 2).astype(np.int32)
        return result


def load_dicom_as_image(dicom_path):
    """
    Load a DICOM file and correct channel order if needed.

    Args:
        dicom_path (str): Path to the DICOM file.

    Returns:
        image (numpy.ndarray): The DICOM image as a NumPy array.
    """
    # Read the DICOM image using SimpleITK
    dicom_image = sitk.ReadImage(dicom_path)

    # Convert the image to a NumPy array
    image_array = sitk.GetArrayFromImage(dicom_image)[0]  # Extract the first slice

    # Normalize the pixel values to 0-255
    image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Check if the image has color channels
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Swap channels to convert RGB to BGR (if needed for OpenCV)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    return image_array

if __name__ == "__main__":
    # Read the image
    full_image_path = r"KD/New/KD-3-dengjinyi/KD-3-dengjinyi/3-4LAD原始.dcm"  # Replace with your grayscale image path
    img = load_dicom_as_image(full_image_path)

    # Use a sample contour for demonstration
    contour = np.array([[[100, 100]], [[200, 100]], [[200, 200]], [[100, 200]]], dtype=np.int32)
    cc = ContourTrimmer(img, contour)
    cc.trim_contour()
