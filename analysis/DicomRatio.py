import cv2
import numpy as np
import SimpleITK as sitk
import pydicom


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
        """
        根据单位代码校正物理间距到毫米(mm)
        DICOM单位代码含义：
        1: 米 (m) -> 需要转换为mm
        2: 厘米 (cm) -> 需要转换为mm  
        3: 英寸 (inches) -> 需要转换为mm 这里实际似乎为cm
        4: 毫米 (mm) -> 不需要转换
        """
        if unit_code == 1:      # 米 -> 毫米
            return delta * 1000.0
        elif unit_code == 2:    # 厘米 -> 毫米
            return delta * 10.0
        elif unit_code == 3:    
            return delta * 10.0 # Treat inches as mm
        elif unit_code == 4:    # 毫米，不需要转换
            return delta
        else:                   # 未知单位，假设为毫米
            return delta

    spacing_x_mm = correct_spacing(delta_x, unit_code_x)
    spacing_y_mm = correct_spacing(delta_y, unit_code_y)

    return spacing_x_mm, spacing_y_mm

if __name__ == "__main__":
    dcm_file = r"F:\medicalimg0815\medicallmg\KD\New\KD-2-chentianxiang\KD-2-chentianxiang\2-1LCA测量.dcm"
    # dcm_file = r"F:\medicalimg0815\0828data\2025.8(2).29-KD+冠脉\GEMS_IMG\2025_AUG\28\GH114157\P8SCBO02"
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