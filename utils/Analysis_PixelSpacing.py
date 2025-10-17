import pydicom
from pydicom.errors import InvalidDicomError
import os

def get_pixel_spacing_from_dcm(dcm_file_path):
    """
    从DICOM文件中读取像素间距(Pixel Spacing)。

    参数:
    dcm_file_path (str): .dcm文件的路径。

    返回:
    tuple or None: 如果成功，返回一个包含两个浮点数的元组 (row_spacing, column_spacing)，
                   单位通常是毫米/像素。如果文件不存在、不是有效的DICOM文件或缺少该标签，则返回 None。
    """
    try:
        # 使用 pydicom 读取文件
        dataset = pydicom.dcmread(dcm_file_path)
        # 访问 'PhysicalDeltaX' 标签 (0028, 0030)
        if 'PhysicalDeltaX' in dataset:
            pixel_spacing = dataset.get("PhysicalDeltaX", 0)
            # 'PixelSpacing' 是一个多值(Multi-value)字段，包含两个值
            # 第一个是行间距（垂直方向），第二个是列间距（水平方向）
            # 它们的值是字符串，需要转换为浮点数
            row_spacing = float(pixel_spacing[0])
            column_spacing = float(pixel_spacing[1])
            
            # 通常行间距和列间距是相等的
            return (row_spacing, column_spacing)
        # # 访问 'PixelSpacing' 标签 (0028, 0030)
        # if 'PixelSpacing' in dataset:
        #     pixel_spacing = dataset.PixelSpacing
        #     # 'PixelSpacing' 是一个多值(Multi-value)字段，包含两个值
        #     # 第一个是行间距（垂直方向），第二个是列间距（水平方向）
        #     # 它们的值是字符串，需要转换为浮点数
        #     row_spacing = float(pixel_spacing[0])
        #     column_spacing = float(pixel_spacing[1])
            
        #     # 通常行间距和列间距是相等的
        #     return (row_spacing, column_spacing)
        else:
            print(f"警告: 文件 '{dcm_file_path}' 中未找到 'PhysicalDeltaX'字段。")
            return None

    except FileNotFoundError:
        print(f"错误: 文件未找到 '{dcm_file_path}'")
        return None
    except InvalidDicomError:
        print(f"错误: '{dcm_file_path}' 不是一个有效的DICOM文件。")
        return None
    except Exception as e:
        print(f"读取文件时发生未知错误: {e}")
        return None

if __name__ == '__main__':
    # --- 使用示例 ---
    # 请将 'your_vascular_image.dcm' 替换为您的DICOM文件实际路径
    path = os.path.dirname(__file__)

    for i in os.listdir(r'Medicallmg/KD/New/KD-3-dengjinyi/KD-3-dengjinyi'):
        
        dcm_path = r'Medicallmg/KD/New/KD-3-dengjinyi/KD-3-dengjinyi/'+i
        spacing = get_pixel_spacing_from_dcm(dcm_path)

        if spacing:
            print(f"成功读取到像素间距！")
            print(f"行间距 (垂直方向): {spacing[0]:.4f} mm/pixel")
            print(f"列间距 (水平方向): {spacing[1]:.4f} mm/pixel")
            
            # 假设我们之前计算出的像素直径是 35.5 pixels
            diameter_in_pixels = 35.5
            
            # 使用列间距（或行间距，如果它们相等）来转换
            # 在大多数情况下，这两个值是相等的，可以任取其一
            pixel_width = spacing[1] 
            diameter_in_mm = diameter_in_pixels * pixel_width
            
            print("-" * 30)
            print(f"如果血管直径为 {diameter_in_pixels} 像素,")
            print(f"那么其实际物理尺寸为: {diameter_in_mm:.2f} mm")