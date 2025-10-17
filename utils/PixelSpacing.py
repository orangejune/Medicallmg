import pydicom
import pydicom.errors
from pydicom.datadict import tag_for_keyword
import logging
import os

# 配置一个简单的日志记录器，用于捕获警告信息
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- 使用关键字定义DICOM标签，提高可读性 ---
# 避免在代码中直接使用 (0x0018, 0x6011) 这样的“魔法数字”
TAG_ULTRASOUND_REGION_SEQUENCE = tag_for_keyword('UltrasoundRegionSequence')
TAG_PHYSICAL_UNITS_X_DIRECTION = tag_for_keyword('PhysicalUnitsXDirection')
TAG_PHYSICAL_UNITS_Y_DIRECTION = tag_for_keyword('PhysicalUnitsYDirection')
TAG_PHYSICAL_DELTA_X = tag_for_keyword('PhysicalDeltaX')
TAG_PHYSICAL_DELTA_Y = tag_for_keyword('PhysicalDeltaY')


def _convert_to_mm(delta: float, unit_code: int) -> float:
    """
    根据单位代码将物理增量值转换为毫米(mm)。

    该函数包含一个针对特定设备错误的特殊处理逻辑。

    Args:
        delta (float): 原始物理增量值。
        unit_code (int): 单位代码。根据DICOM标准和常见实践：
                         1: 厘米 (cm)
                         2: 毫米 (mm)
                         3: 英寸 (inches) - 在此函数中被特殊处理

    Returns:
        float: 转换为毫米后的值。
    """
    if unit_code == 3:
        # --- 核心修正逻辑 ---
        # 特殊情况：处理一个已知的设备错误。
        # 该设备错误地将单位标记为3 (英寸)，但实际数值是以厘米(cm)为单位的。
        # 因此，我们按厘米(cm)到毫米(mm)进行转换。
        logging.warning(
            "Unit code is 3 (inches), but treating the value as centimeters "
            "due to a known data inconsistency. Converting from cm to mm."
        )
        return delta * 10.0  # 1 cm = 10 mm
    elif unit_code == 1:  # 标准情况：单位是厘米 (cm)
        return delta * 10.0  # 1 cm = 10 mm
    elif unit_code == 2:  # 标准情况：单位是毫米 (mm)
        return delta  # 无需转换
    else:
        # 对于未知或不支持的单位代码，记录警告并返回原始值
        logging.warning(
            f"Unknown or unsupported unit code '{unit_code}'. "
            "Returning the original delta value without conversion."
        )
        return delta


def get_corrected_pixel_spacing(dicom_path: str) -> tuple[float | None, float | None]:
    """
    从DICOM文件的超声区域序列(Ultrasound Region Sequence)中提取并修正像素间距。

    此函数专门设计用于处理一种特殊情况：当单位代码被错误地标记为3（英寸）时，
    函数会假定其数值单位实为厘米（cm）并进行相应转换。
    同时，它也能正确处理标准的厘米和毫米单位。

    Args:
        dicom_path (str): DICOM文件的路径。

    Returns:
        tuple[float | None, float | None]: 一个包含X和Y方向像素间距（单位：毫米）的元组。
                                             如果无法找到或计算间距，则返回 (None, None)。
    """
    # --- 1. 读取文件并进行基本校验 ---
    try:
        ds = pydicom.dcmread(dicom_path)
    except FileNotFoundError:
        logging.error(f"DICOM file not found at path: {dicom_path}")
        return None, None
    except pydicom.errors.InvalidDicomError:
        logging.error(f"The file at {dicom_path} is not a valid DICOM file.")
        return None, None

    # --- 2. 安全地获取超声区域序列 ---
    # 使用 .get() 方法，如果标签不存在，会返回None，避免程序崩溃。
    region_element = ds.get(TAG_ULTRASOUND_REGION_SEQUENCE, None)
    if not region_element:
        logging.info(f"'{dicom_path}' does not contain Ultrasound Region Sequence.")
        return None, None

    # 序列的值是一个列表，可能为空
    region_seq = region_element.value
    if not region_seq or len(region_seq) == 0:
        logging.info("Ultrasound Region Sequence is present but empty.")
        return None, None

    # 假设我们只关心序列中的第一个区域项目。
    # 在某些复杂情况下，可能需要遍历所有项目或根据其他标准选择。
    item = region_seq[0]

    # --- 3. 从序列项目中提取间距和单位信息 ---
    delta_x_elem = item.get(TAG_PHYSICAL_DELTA_X, None)
    delta_y_elem = item.get(TAG_PHYSICAL_DELTA_Y, None)

    # 物理间距值是必需的
    if delta_x_elem is None or delta_y_elem is None:
        logging.warning("PhysicalDeltaX or PhysicalDeltaY is missing in the sequence item.")
        return None, None
    
    # 单位代码是可选的，如果缺失，根据常见实践默认其为1 (cm)
    unit_code_x_elem = item.get(TAG_PHYSICAL_UNITS_X_DIRECTION, None)
    unit_code_y_elem = item.get(TAG_PHYSICAL_UNITS_Y_DIRECTION, None)

    # --- 4. 转换值为正确的类型 ---
    delta_x = float(delta_x_elem.value)
    delta_y = float(delta_y_elem.value)
    
    # 如果单位代码不存在，默认设为 1 (cm)，这是一个安全的行业假设
    unit_code_x = int(unit_code_x_elem.value) if unit_code_x_elem else 1
    unit_code_y = int(unit_code_y_elem.value) if unit_code_y_elem else 1

    # --- 5. 应用转换和修正逻辑 ---
    spacing_x_mm = _convert_to_mm(delta_x, unit_code_x)
    spacing_y_mm = _convert_to_mm(delta_y, unit_code_y)

    return spacing_x_mm, spacing_y_mm