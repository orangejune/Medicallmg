import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev
# import pims
# import tifffile
import SimpleITK as sitk

class Analysis_ReadImg:
    def __init__(self, full_image_path, blue_image_path, is_dicom=False):
        """
        Initialize the ContourMaskProcessor class.

        Args:
            full_image_path (str): Path to the full image (DICOM or other formats).
            blue_image_path (str): Path to the blue-labeled image (DICOM or other formats).
            is_dicom (bool): Set to True if the input files are DICOM (.dcm) files.
        """
        self.full_image_path = full_image_path
        self.blue_image_path = blue_image_path

        if is_dicom:
            # Load DICOM files
            self.full_image = self._load_dicom_as_image(full_image_path)
            self.blue_image = self._load_dicom_as_image(blue_image_path)
        else:
            # Load standard image formats
            self.full_image = cv2.imread(full_image_path)
            self.blue_image = cv2.imread(blue_image_path)

        if self.full_image is None or self.blue_image is None:
            raise FileNotFoundError("Unable to load one or both images.")

        # Debug the loaded images
        # self.debug_image_channels(self.full_image, "Full Image")
        # self.debug_image_channels(self.blue_image, "Blue Image with Labels")

        # Dictionary to store masks and buffers for later analysis
        self.masks = {}
        self.buffers = {}
        self.restricted_contours = []


    def debug_image_channels(self, image, image_name="Image"):
        """
        Debug the channel order of an image and verify pixel values.

        Args:
            image (numpy.ndarray): The image to debug.
            image_name (str): Name for debugging logs.
        """
        if image is None:
            print(f"{image_name} is not loaded.")
            return

        print(f"{image_name} Shape: {image.shape}")
        if len(image.shape) == 3 and image.shape[2] == 3:
            print(f"{image_name} Pixel at (100, 100): {image[100, 100]} (Format: BGR)")
            # Split channels
            b, g, r = cv2.split(image)
            # print(f"{image_name} - Blue Max: {np.max(b)}")
            # print(f"{image_name} - Green Max: {np.max(g)}")
            # print(f"{image_name} - Red Max: {np.max(r)}")
            #
            # # Visualize channels
            # cv2.imshow(f"{image_name} - Blue Channel", b)
            # cv2.imshow(f"{image_name} - Green Channel", g)
            # cv2.imshow(f"{image_name} - Red Channel", r)
        else:
            print(f"{image_name} is grayscale or has an unexpected format.")

        cv2.imshow(image_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def debug_dicom_metadata(self, dicom_path):
        """
        Debug the metadata of a DICOM file.

        Args:
            dicom_path (str): Path to the DICOM file.
        """
        dicom_image = sitk.ReadImage(dicom_path)
        print(f"Loaded DICOM Metadata for {dicom_path}:")
        for key in dicom_image.GetMetaDataKeys():
            print(f"{key}: {dicom_image.GetMetaData(key)}")


    def _load_dicom_as_image(self, dicom_path):
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

