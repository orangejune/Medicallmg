import cv2
import numpy as np
import SimpleITK as sitk

def detect_lines_hough(img):
    """
    Automatically convert to actual length based on the scale

    Returns:
        list: A list of tuples representing the (x, y) coordinates of bright points.
    """
    # img = cv2.imread(img)
    image = img.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]

    edges = cv2.Canny(gray[:600,:800], 50, 150, apertureSize=3)

    # Apply the Hough Line Transform to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    bright_points = []
    brightness_threshold = 30

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            if 0 < theta * (180 / np.pi) < 90:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * (rho - 1)  # Shift the line so that it intersects with the scale marks
                y0 = b * (rho - 1)
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
                # cv2.imshow('Hough Lines', image)
                # cv2.waitKey(0)

                points_on_line = np.linspace((x1, y1), (x2, y2), num=max(width, height), dtype=int)

                for point in points_on_line:
                    x, y = point
                    if 0 <= x < width and 0 <= y < 600:  # Ignore the bottom curve
                        brightness = gray[y, x]
                        # Check if the brightness exceeds the threshold
                        if brightness > brightness_threshold:
                            bright_points.append((x, y))

    # Merge points that are close to each other vertically
    merged_points = []
    current_group = []
    for i, point in enumerate(bright_points):
        if i == 0:
            current_group.append(point)
        else:
            prev_point = bright_points[i - 1]
            # Check if the vertical distance between points is less than 3 pixels
            if abs(point[1] - prev_point[1]) < 3:
                current_group.append(point)
            else:
                if current_group:
                    # Calculate the average x and y coordinates of the group
                    avg_x = int(np.mean([p[0] for p in current_group]))
                    avg_y = int(np.mean([p[1] for p in current_group]))
                    merged_points.append((avg_x, avg_y))
                    current_group = [point]
                else:
                    current_group.append(point)

    # Handle the last group of points
    if current_group:
        avg_x = int(np.mean([p[0] for p in current_group]))
        avg_y = int(np.mean([p[1] for p in current_group]))
        merged_points.append((avg_x, avg_y))

    return merged_points

def cal_distance_ratio(img):
    '''
        实际1mm对应图像多少像素
        diameter_in_mm = max_diameter_in_pixel / ratio
    '''
    bright_points = detect_lines_hough(img)
    # for i in bright_points:
    #     cv2.circle(img, i, 3, (255, 255, 0), -1)
    # cv2.imshow('Result', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    pixel_distance = []
    for i in range(len(bright_points)-1):
        pixel_distance.append(np.linalg.norm(np.array(bright_points[i])-np.array(bright_points[i+1])))
    ratio = np.mean(pixel_distance)/10
    return ratio
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
    # full_image_path = r"F:\medicalimg0815\0828data\2025.8(2).29-KD+冠脉\GEMS_IMG\2025_AUG\28\GH114157\P8SCBL1G"  # Replace with your grayscale image path
    full_image_path = r"F:\medicalimg0815\medicallmg\KD\New\KD-2-chentianxiang\KD-2-chentianxiang\2-1LCA测量.dcm"  # Replace with your grayscale image path
    # full_image_path = r"KD/New/冠脉瘤1-wangziyi/KD-CAL-wangziyi/1-左冠脉整体.dcm"  # Replace with your grayscale image path

    img = load_dicom_as_image(full_image_path)

    ratio = cal_distance_ratio(img)
    print(f'ratio:{ratio}')