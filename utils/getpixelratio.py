import cv2
import numpy as np
import SimpleITK as sitk
import pydicom

def detect_lines_hough(img):
    """
    Automatically convert to actual length based on the scale


    Args:
        img (numpy.ndarray): The input image in BGR format.

    Returns:
        list: A list of tuples representing the (x, y) coordinates of bright points.
    """
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
                # cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
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

# if __name__ == "__main__":
#     full_image_path = r"KD/New/KD-3-dengjinyi/KD-3-dengjinyi/3-4LAD原始.dcm"  # Replace with your grayscale image path
#     # full_image_path = r"KD/New/KD-2-chentianxiang/KD-2-chentianxiang/2-8LAD原始.dcm"  # Replace with your grayscale image path
#     # full_image_path = r"KD/New/冠脉瘤1-wangziyi/KD-CAL-wangziyi/1-左冠脉整体.dcm"  # Replace with your grayscale image path
#
#     img = load_dicom_as_image(full_image_path)
#
#     ratio = cal_distance_ratio(img)
#     print(f'ratio:{ratio}')




##todo: part 2
def explore_ultrasound_region_attributes(dicom_path):
    """
    Load the DICOM file and print all attributes inside the Ultrasound Region Sequence (0018,6011).
    """
    ds = pydicom.dcmread(dicom_path)

    region_seq = ds.get((0x0018, 0x6011), None)
    if not region_seq:
        print("Ultrasound Region Sequence (0018,6011) not found.")
        return

    for i, item in enumerate(region_seq):
        print(f"\n--- Region Item {i} ---")
        for elem in item:
            tag = elem.tag
            name = elem.name
            value = elem.value
            print(f"{tag}  {name:<40} {value}")


def get_corrected_pixel_spacing(dicom_path):
    """
    Extract corrected pixel spacing from the Ultrasound Region Sequence,
    overriding unit code 3 (inches) to assume centimeters.
    Returns (spacing_x_mm, spacing_y_mm)
    """
    ds = pydicom.dcmread(dicom_path)

    region_element = ds.get((0x0018, 0x6011), None)
    if not region_element:
        return None, None

    region_seq = region_element.value
    if not region_seq or len(region_seq) == 0:
        return None, None

    item = region_seq[0]

    delta_x_elem = item.get((0x0018, 0x602C), None)
    delta_y_elem = item.get((0x0018, 0x602E), None)
    unit_code_x_elem = item.get((0x0018, 0x6024), None)
    unit_code_y_elem = item.get((0x0018, 0x6026), None)

    if delta_x_elem is None or delta_y_elem is None:
        return None, None

    delta_x = float(delta_x_elem.value)
    delta_y = float(delta_y_elem.value)
    unit_code_x = int(unit_code_x_elem.value) if unit_code_x_elem else 1
    unit_code_y = int(unit_code_y_elem.value) if unit_code_y_elem else 1

    def correct_spacing(delta, unit_code):
        if unit_code == 3:
            return delta * 100.0  # Treat inches as cm
        # elif unit_code == 1:
        #     return delta
        # elif unit_code == 2:
        #     return delta * 10.0
        # elif unit_code == 4:
        #     return delta * 1000.0
        # else:
        #     return delta

    spacing_x_mm = correct_spacing(delta_x, unit_code_x)
    spacing_y_mm = correct_spacing(delta_y, unit_code_y)

    return spacing_x_mm, spacing_y_mm


dcm_file = r"Medicallmg/KD/New/KD-3-dengjinyi/KD-3-dengjinyi/3-4LAD原始.dcm"
# explore_ultrasound_region_attributes(dcm_file)

spacing_x, spacing_y = get_corrected_pixel_spacing(dcm_file)
if spacing_x:
    print(f"Corrected spacing X: {spacing_x:.4f} mm/pixel")
    print(f"Corrected spacing Y: {spacing_y:.4f} mm/pixel")
    pixel_dist = 100  # Example
    print(f"Real-world distance for 100 px: {pixel_dist * spacing_x:.2f} mm")
else:
    print("Could not extract spacing.")


##  todo: use the info    (0018,602C)	Physical DeltaX	0.00703049