import pydicom
from pydicom.dicomdir import DicomDir
import os

import pydicom
import json
import pydicom

import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pydicom
import cv2
import numpy as np


def serialize_value(value):
    """
    Convert non-JSON serializable values into a string or a simpler representation.
    """
    if isinstance(value, pydicom.valuerep.PersonName):
        return str(value)  # Convert PersonName to a string
    if isinstance(value, pydicom.uid.UID):
        return str(value)  # Convert UID to a string
    if isinstance(value, list):  # Handle lists (e.g., ImageOrientationPatient)
        return [serialize_value(v) for v in value]
    return value  # Return other types as-is


def extract_dicomdir_info(dicomdir_path):
    """
    Extract detailed information from a DICOMDIR file.

    Parameters:
        dicomdir_path (str): Path to the DICOMDIR file.

    Returns:
        list: A list of dictionaries representing patients, studies, series, and instances.
    """
    dicomdir = pydicom.dcmread(dicomdir_path)
    data = []

    for record in dicomdir.DirectoryRecordSequence:
        if record.DirectoryRecordType == "PATIENT":
            patient = {
                "PatientID": serialize_value(getattr(record, "PatientID", "Unknown")),
                "PatientName": serialize_value(getattr(record, "PatientName", "Unknown")),
                "PatientSex": serialize_value(getattr(record, "PatientSex", "Unknown")),
                "Studies": []
            }
            data.append(patient)
        elif record.DirectoryRecordType == "STUDY":
            study = {
                "StudyID": serialize_value(getattr(record, "StudyID", "Unknown")),
                "StudyInstanceUID": serialize_value(getattr(record, "StudyInstanceUID", "Unknown")),
                "StudyDate": serialize_value(getattr(record, "StudyDate", "Unknown")),
                "StudyDescription": serialize_value(getattr(record, "StudyDescription", "Unknown")),
                "AccessionNumber": serialize_value(getattr(record, "AccessionNumber", "Unknown")),
                "Series": []
            }
            data[-1]["Studies"].append(study)
        elif record.DirectoryRecordType == "SERIES":
            series = {
                "SeriesInstanceUID": serialize_value(getattr(record, "SeriesInstanceUID", "Unknown")),
                "SeriesNumber": serialize_value(getattr(record, "SeriesNumber", "Unknown")),
                "Modality": serialize_value(getattr(record, "Modality", "Unknown")),
                "SeriesDescription": serialize_value(getattr(record, "SeriesDescription", "Unknown")),
                "BodyPartExamined": serialize_value(getattr(record, "BodyPartExamined", "Unknown")),
                "Instances": []
            }
            data[-1]["Studies"][-1]["Series"].append(series)
        elif record.DirectoryRecordType == "IMAGE":
            image = {
                "SOPInstanceUID": serialize_value(getattr(record, "ReferencedSOPInstanceUIDInFile", "Unknown")),
                "ReferencedFileID": serialize_value(
                    "\\".join(record.ReferencedFileID) if hasattr(record, "ReferencedFileID") else "Unknown"),
                "InstanceNumber": serialize_value(getattr(record, "InstanceNumber", "Unknown")),
                "ImagePositionPatient": serialize_value(getattr(record, "ImagePositionPatient", "Unknown")),
                "ImageOrientationPatient": serialize_value(getattr(record, "ImageOrientationPatient", "Unknown"))
            }
            data[-1]["Studies"][-1]["Series"][-1]["Instances"].append(image)

    return data

##todo: classify view
def classify_view_from_image_v2(dicom_path):
    """
    Classify the view type based on the pixel content of the DICOM image.
    """
    dicom = pydicom.dcmread(dicom_path)
    pixel_array = dicom.pixel_array

    # Normalize the pixel data to [0, 255]
    image = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply edge detection to highlight structures
    edges = cv2.Canny(image, threshold1=50, threshold2=150)

    # Use simple heuristics (placeholder logic for illustrative purposes)
    if np.mean(edges) > 20:
        return "Likely Apical View"
    elif np.mean(edges) <= 20:
        return "Likely Parasternal or Other View"
    else:
        return "Unknown"


##todo: infer view
def infer_view_from_metadata(dicom):
    """
    Infer view type from DICOM metadata.
    """
    view_type = "Unknown"

    # Check Image Orientation (Patient)
    if "ImageOrientationPatient" in dicom:
        orientation = dicom.ImageOrientationPatient
        if orientation == [1, 0, 0, 0, 1, 0]:
            view_type = "Axial"
        elif orientation == [0, 1, 0, 0, 0, -1]:
            view_type = "Coronal"
        elif orientation == [1, 0, 0, 0, 0, -1]:
            view_type = "Sagittal"
        else:
            view_type = "Custom Orientation"

    # Check Series Description or Protocol Name
    if "SeriesDescription" in dicom:
        desc = dicom.SeriesDescription.lower()
        if "apical" in desc:
            view_type = "Apical"
        elif "parasternal" in desc:
            view_type = "Parasternal"
        elif "subcostal" in desc:
            view_type = "Subcostal"

    if "ProtocolName" in dicom:
        protocol = dicom.ProtocolName.lower()
        if "apical" in protocol:
            view_type = "Apical"
        elif "parasternal" in protocol:
            view_type = "Parasternal"
        elif "subcostal" in protocol:
            view_type = "Subcostal"

    return view_type

def classify_view_from_image(dicom):
    """
    Classify view type based on pixel content.
    """
    pixel_array = dicom.pixel_array

    # Normalize the pixel array to [0, 255]
    image = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Edge detection to identify structures
    edges = cv2.Canny(image, threshold1=50, threshold2=150)

    # Use simple heuristics (this is a placeholder for more advanced logic)
    if np.mean(edges) > 50:
        return "Apical or Subcostal View"
    else:
        return "Parasternal or Other View"

def infer_and_classify_view(dicom_path):
    """
    Combine metadata and image analysis to infer and classify the view.
    """
    dicom = pydicom.dcmread(dicom_path)

    # Infer from metadata
    metadata_view = infer_view_from_metadata(dicom)

    # Classify from image
    image_view = classify_view_from_image(dicom)

    return {
        "MetadataBasedView": metadata_view,
        "ImageBasedView": image_view
    }

def infer_view_from_orientation(dicom_path):
    """
    Infer the view type based on Image Orientation and Position metadata.
    """
    dicom = pydicom.dcmread(dicom_path)
    view_type = "Unknown"

    # Check Image Orientation (Patient)
    if "ImageOrientationPatient" in dicom:
        orientation = dicom.ImageOrientationPatient
        # Infer orientation
        if orientation == [1, 0, 0, 0, 1, 0]:
            view_type = "Axial"
        elif orientation == [0, 1, 0, 0, 0, -1]:
            view_type = "Coronal"
        elif orientation == [1, 0, 0, 0, 0, -1]:
            view_type = "Sagittal"
        else:
            view_type = "Other (Custom Orientation)"

    # Print Image Position (Patient) for additional spatial context
    if "ImagePositionPatient" in dicom:
        position = dicom.ImagePositionPatient
        print(f"Image Position: {position}")

    return view_type

# Path to the DICOMDIR file
dicomdir_path = "KD\KD\DICOMDIR"  # Replace with your file path

# Extract and process information
# dicomdir_info = extract_dicomdir_info(dicomdir_path)
#
# # Save extracted data to a JSON file
# output_file = "DICOMDIR_extracted_info.json"
# with open(output_file, "w") as f:
#     json.dump(dicomdir_info, f, indent=4)
#
# print(f"Extracted information saved to {output_file}")

# Example usage

# dicom_path = r"KD\KD\GEMS_IMG\2024_JUN\06\KW121239\O66C7NOK"  # Replace with your DICOM file path
# result = infer_and_classify_view(dicom_path)
# print(f"Inferred View from Metadata: {result['MetadataBasedView']}")
# print(f"Classified View from Image: {result['ImageBasedView']}")

# Example usage
# dicom_path = "path_to_dicom_file.dcm" --patient info unknown
# view_type = infer_view_from_orientation(dicom_path)
# print(f"Inferred View Type: {view_type}")

# Example usage
# # dicom_path = "path_to_dicom_file.dcm"
# view_type = classify_view_from_image(dicom_path)
# print(f"Inferred View Type from Image: {view_type}")



import pydicom

dicom_file = r"KD\KD\GEMS_IMG\2024_JUN\06\KW121239\O66C6H84"
dicom = pydicom.dcmread(dicom_file)

# Check for Graphic Annotation Sequence
if hasattr(dicom, "GraphicAnnotationSequence"):
    print("\n### Graphic Annotation Sequence ###")
    for i, annotation in enumerate(dicom.GraphicAnnotationSequence):
        print(f"Annotation {i + 1}:")
        if hasattr(annotation, "GraphicLayer"):
            print(f"  - Graphic Layer: {annotation.GraphicLayer}")
        if hasattr(annotation, "GraphicObjectsSequence"):
            for j, graphic_object in enumerate(annotation.GraphicObjectsSequence):
                print(f"    Graphic Object {j + 1}:")
                if hasattr(graphic_object, "GraphicData"):
                    print(f"      - Graphic Data (points): {graphic_object.GraphicData}")
                if hasattr(graphic_object, "BoundingBoxAnnotationUnits"):
                    print(f"      - Bounding Box Units: {graphic_object.BoundingBoxAnnotationUnits}")

# Check for Display Shutter attributes
if hasattr(dicom, "ShutterShape"):
    print("\n### Display Shutter Attributes ###")
    print(f"  - Shutter Shape: {dicom.ShutterShape}")
    if hasattr(dicom, "ShutterLeftVerticalEdge"):
        print(f"  - Left Vertical Edge: {dicom.ShutterLeftVerticalEdge}")
    if hasattr(dicom, "ShutterRightVerticalEdge"):
        print(f"  - Right Vertical Edge: {dicom.ShutterRightVerticalEdge}")
    if hasattr(dicom, "ShutterUpperHorizontalEdge"):
        print(f"  - Upper Horizontal Edge: {dicom.ShutterUpperHorizontalEdge}")
    if hasattr(dicom, "ShutterLowerHorizontalEdge"):
        print(f"  - Lower Horizontal Edge: {dicom.ShutterLowerHorizontalEdge}")

# Check for ROI Contour Sequence
if hasattr(dicom, "ROIContourSequence"):
    print("\n### ROI Contour Sequence ###")
    for i, roi in enumerate(dicom.ROIContourSequence):
        print(f"ROI {i + 1}:")
        if hasattr(roi, "ReferencedROINumber"):
            print(f"  - Referenced ROI Number: {roi.ReferencedROINumber}")
        if hasattr(roi, "ContourSequence"):
            for j, contour in enumerate(roi.ContourSequence):
                print(f"    Contour {j + 1}:")
                if hasattr(contour, "ContourData"):
                    print(f"      - Contour Data (points): {contour.ContourData}")

# Check for Overlay Data
if hasattr(dicom, "OverlayData"):
    print("\n### Overlay Data ###")
    rows = int(dicom.get("OverlayRows"))
    cols = int(dicom.get("OverlayColumns"))
    overlay_data = dicom.OverlayData

    # Convert bit-packed overlay data to binary array
    overlay_bits = np.unpackbits(np.frombuffer(overlay_data, dtype=np.uint8))
    overlay_array = overlay_bits[: rows * cols].reshape(rows, cols)
    print(f"  - Overlay extracted with shape: {overlay_array.shape}")

else:
    print("\nNo overlays, graphic annotations, or ROI contours were found.")
