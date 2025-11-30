from flask import Flask, request, jsonify, render_template
import os
import subprocess
import json
from datetime import datetime
import sys
import importlib.util
import cv2
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import json


# 添加当前项目路径到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/')
def index():
    return render_template('index.html')

def clear_static_directories():
    """
    清空viewer/static目录下的临时文件和结果文件
    """
    directories_to_clear = [
        'viewer/static/images',
        'viewer/static/temp',
        'viewer/static/contours',
        'viewer/static/results'
    ]
    
    for directory in directories_to_clear:
        if os.path.exists(directory):
            # 删除目录中的所有文件
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"无法删除文件 {file_path}: {e}")

@app.route('/process-dicom', methods=['POST'])
def process_dicom():
    try:
        # 在处理新文件前清空静态目录
        clear_static_directories()

        file = request.files['file']
        if not file:
            return jsonify({'error': '未上传文件'}), 400
        
        # 保存上传的文件
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)

        # 图片保存地址
        output_folder = f'viewer/static/images'
        os.makedirs(output_folder, exist_ok=True)

        # 添加 processing.py 所在的父目录到 Python 路径
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(parent_dir)

        from processing import process_dicom_to_frames
        success, pixel_spacing, unit = process_dicom_to_frames(file_path, output_folder)

        if success:
            # 获取所有生成的帧文件名
            frame_files = [f for f in os.listdir(output_folder) if f.endswith('.jpg')]
            frame_files.sort()  # 按顺序排列

            return jsonify({
                'success': True,
                'message': 'DICOM 转换成功',
                'frames': frame_files,
                'output_folder': output_folder,
                'pixel_spacing': pixel_spacing,
                'unit': unit
            })
        else:
            return jsonify({
                'error': 'DICOM 转换失败'
            }), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-results', methods=['GET'])
def get_results():
    try:
        results = []
        for filename in os.listdir('results'):
            if filename.endswith('.jpg'):
                results.append({
                    'name': filename,
                    'path': f'/static/results/{filename}',
                    'score': 0.85
                })
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 获取ROI并识别边界测量直径
@app.route('/measure-frame', methods=['POST'])
def measure_frame():
    try:
        data = request.get_json()
        frame_name = data.get('frame_name')
        
        if not frame_name:
            return jsonify({'error': '缺少帧名称'}), 400
            
        # 构建帧文件路径
        image_path = os.path.join('viewer/static/images', frame_name)
        
        if not os.path.exists(image_path):
            return jsonify({'error': '图像文件不存在'}), 404
            
        # 添加 processing.py 所在的父目录到 Python 路径
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(parent_dir)
        
        import processing
        
        # 使用YOLO模型进行预测
        results = processing.yolo_model.predict(source=image_path, save=False, imgsz=640, device='cpu')
        
        boxes_info = []
        contour_info = []
        
        if results and len(results) > 0:
            for r in results:
                if hasattr(r, 'boxes') and r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cls = int(box.cls[0])
                        conf = float(box.conf.item())
                        
                        # 保存边界框信息
                        boxes_info.append({
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2,
                            'class': cls,
                            'confidence': conf
                        })
                        
                        # 裁剪 ROI 区域
                        original_img = cv2.imread(image_path)
                        roi = original_img[int(y1):int(y2), int(x1):int(x2)]
                        
                        # 保存 ROI 临时文件
                        roi_temp_path = os.path.join('viewer/static/temp', f'roi_{frame_name}')
                        os.makedirs(os.path.dirname(roi_temp_path), exist_ok=True)
                        cv2.imwrite(roi_temp_path, roi)
                        
                        # 使用 U-Net 进行边界识别
                        contour_save_path = os.path.join('viewer/static/contours', f'contour_{frame_name}')
                        os.makedirs(os.path.dirname(contour_save_path), exist_ok=True)
                        
                        result_img, contour, final_score, largest_lumen_mask = processing.predict_contour_and_save(roi_temp_path, contour_save_path)
                        
                        if contour is not None:
                            # 计算最大直径并生成可视化图片
                            max_diameter_value_in_pixel = 0
                            max_diameter_image_path = None
                            p1_int = [0, 0]
                            p2_int = [0, 0]
                            
                            try:
                                # 创建用于最大直径可视化的保存路径
                                max_diameter_save_path = os.path.join('viewer/static/results', f'max_diameter_contour_{frame_name}')
                                os.makedirs(os.path.dirname(max_diameter_save_path), exist_ok=True)
                                
                                # 创建二值掩码用于计算最大直径
                                lumen_binary_mask = np.zeros_like(roi[:, :, 0])  # 假设roi是彩色图像
                                cv2.drawContours(lumen_binary_mask, [contour], -1, 255, cv2.FILLED)
                                
                                # 计算最大直径
                                max_diameter_value_in_pixel, p1_int, p2_int = processing.find_and_visualize_max_diameter_improved(
                                    largest_lumen_mask,
                                    contour_save_path,
                                    max_diameter_save_path
                                )
                                
                                # 设置返回的图片路径
                                max_diameter_image_path = f'/static/results/max_diameter_contour_{frame_name}'
                            except Exception as e:
                                print(f"计算最大直径时出错: {e}")
                            
                            # 保存测量结果为JSON文件
                            measurement_data = {
                                'contour_image': f'/static/contours/contour_{frame_name}',
                                'max_diameter_image_path': max_diameter_image_path,
                                'score': float(final_score),
                                'max_diameter_in_pixel': float(max_diameter_value_in_pixel),
                                'max_diameter_image': max_diameter_image_path,
                                'box': {
                                    'x1': float(x1),
                                    'y1': float(y1),
                                    'x2': float(x2),
                                    'y2': float(y2)
                                },
                                'line_points': {
                                    'p1': {
                                        'x': int(p1_int[0]),
                                        'y': int(p1_int[1])
                                    },
                                    'p2': {
                                        'x': int(p2_int[0]),
                                        'y': int(p2_int[1])
                                    }
                                }
                            }
                            
                            # 保存为JSON文件
                            json_save_path = os.path.join('viewer/static/results', f'measurement_{frame_name.replace(".jpg", "")}.json')
                            with open(json_save_path, 'w') as f:
                                json.dump(measurement_data, f)
                            
                            contour_info.append(measurement_data)
        
        return jsonify({
            'success': True,
            'boxes': boxes_info,
            'contours': contour_info
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 批量测量
@app.route('/batch-measure', methods=['POST'])
def batch_measure():
    try:
        data = request.get_json()
        frame_names = data.get('frame_names', [])
        
        if not frame_names:
            return jsonify({'error': '缺少帧名称列表'}), 400
            
        # 存储所有测量结果
        all_contours = []
        
        # 处理每一帧
        for frame_name in frame_names:
            # 构建帧文件路径
            image_path = os.path.join('viewer/static/images', frame_name)
            
            if not os.path.exists(image_path):
                continue
                
            # 添加 processing.py 所在的父目录到 Python 路径
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.append(parent_dir)
            
            import processing
            
            # 使用YOLO模型进行预测
            results = processing.yolo_model.predict(source=image_path, save=False, imgsz=640, device='cpu')
            
            frame_contours = []
            
            if results and len(results) > 0:
                for r in results:
                    if hasattr(r, 'boxes') and r.boxes is not None:
                        for box in r.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            cls = int(box.cls[0])
                            conf = float(box.conf.item())
                            
                            # 裁剪 ROI 区域
                            original_img = cv2.imread(image_path)
                            roi = original_img[int(y1):int(y2), int(x1):int(x2)]
                            
                            # 保存 ROI 临时文件
                            roi_temp_path = os.path.join('viewer/static/temp', f'roi_{frame_name}')
                            os.makedirs(os.path.dirname(roi_temp_path), exist_ok=True)
                            cv2.imwrite(roi_temp_path, roi)
                            
                            # 使用 U-Net 进行边界识别
                            contour_save_path = os.path.join('viewer/static/contours', f'contour_{frame_name}')
                            os.makedirs(os.path.dirname(contour_save_path), exist_ok=True)
                            
                            result_img, contour, final_score, largest_lumen_mask = processing.predict_contour_and_save(roi_temp_path, contour_save_path)
                            
                            if contour is not None:
                                # 计算最大直径并生成可视化图片
                                max_diameter_value_in_pixel = 0
                                max_diameter_image_path = None
                                p1_int = [0, 0]
                                p2_int = [0, 0]
                                
                                try:
                                    # 创建用于最大直径可视化的保存路径
                                    max_diameter_save_path = os.path.join('viewer/static/results', f'max_diameter_contour_{frame_name}')
                                    os.makedirs(os.path.dirname(max_diameter_save_path), exist_ok=True)
                                    
                                    # 创建二值掩码用于计算最大直径
                                    lumen_binary_mask = np.zeros_like(roi[:, :, 0])  # 假设roi是彩色图像
                                    cv2.drawContours(lumen_binary_mask, [contour], -1, 255, cv2.FILLED)
                                    
                                    # 计算最大直径
                                    max_diameter_value_in_pixel, p1_int, p2_int = processing.find_and_visualize_max_diameter_improved(
                                        largest_lumen_mask,
                                        contour_save_path,
                                        max_diameter_save_path
                                    )
                                    
                                    # 设置返回的图片路径
                                    max_diameter_image_path = f'/static/results/max_diameter_contour_{frame_name}'
                                except Exception as e:
                                    print(f"计算最大直径时出错: {e}")

                                # 保存测量结果为JSON文件
                                measurement_data = {
                                    'frame_name': frame_name,
                                    'contour_image': f'/static/contours/contour_{frame_name}',
                                    'max_diameter_image_path': max_diameter_image_path,
                                    'score': float(final_score),
                                    'max_diameter_in_pixel': float(max_diameter_value_in_pixel),
                                    'max_diameter_image': max_diameter_image_path,
                                    'box': {
                                        'x1': float(x1),
                                        'y1': float(y1),
                                        'x2': float(x2),
                                        'y2': float(y2)
                                    },
                                    'line_points': {
                                        'p1': {
                                            'x': int(p1_int[0]),
                                            'y': int(p1_int[1])
                                        },
                                        'p2': {
                                            'x': int(p2_int[0]),
                                            'y': int(p2_int[1])
                                        }
                                    }
                                }
                                
                                # 保存为JSON文件
                                json_save_path = os.path.join('viewer/static/results', f'measurement_{frame_name.replace(".jpg", "")}.json')
                                with open(json_save_path, 'w') as f:
                                    json.dump(measurement_data, f)
                                
                                frame_contours.append(measurement_data)
            
            all_contours.extend(frame_contours)
        
        return jsonify({
            'success': True,
            'contours': all_contours
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/get-measurement-result', methods=['GET'])
def get_measurement_result():
    try:
        frame_name = request.args.get('frame_name')
        if not frame_name:
            return jsonify({'error': '缺少帧名称'}), 400
            
        json_path = os.path.join('viewer/static/results', f'measurement_{frame_name.replace(".jpg", "")}.json')
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                measurement_data = json.load(f)
            return jsonify(measurement_data)
        else:
            return jsonify({'error': '未找到测量结果'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# roi框选测量
@app.route('/roi-measure', methods=['POST'])
def roi_measure():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '无效的请求数据'}), 400
            
        frame_name = data.get('frame_name')
        roi = data.get('roi')  # {x1, y1, x2, y2}
        
        if not frame_name or not roi:
            return jsonify({'error': '缺少帧名称或ROI信息'}), 400
            
        # 构建帧文件路径
        image_path = os.path.join('viewer/static/images', frame_name)
        
        if not os.path.exists(image_path):
            return jsonify({'error': '图像文件不存在'}), 404
            
        # 添加 processing.py 所在的父目录到 Python 路径
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(parent_dir)
        
        import processing
        
        # 读取原始图像
        original_img = cv2.imread(image_path)
        if original_img is None:
            return jsonify({'error': '无法读取图像文件'}), 500
        
        # 提取ROI区域
        x1 = int(roi['x1'])
        y1 = int(roi['y1'])
        x2 = int(roi['x2'])
        y2 = int(roi['y2'])
        
        # 确保坐标在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(original_img.shape[1], x2)
        y2 = min(original_img.shape[0], y2)
        
        # 检查ROI区域是否有效
        if x2 <= x1 or y2 <= y1:
            return jsonify({'error': '无效的ROI区域'}), 400
            
        roi_img = original_img[y1:y2, x1:x2]
        
        # 生成唯一的时间戳用于文件名，避免重复
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        
        # 保存 ROI 临时文件（使用时间戳确保唯一性）
        roi_temp_path = os.path.join('viewer/static/temp', f'roi_manual_{timestamp}_{frame_name}')
        os.makedirs(os.path.dirname(roi_temp_path), exist_ok=True)
        cv2.imwrite(roi_temp_path, roi_img)
        
        # 使用 U-Net 进行边界识别
        contour_save_path = os.path.join('viewer/static/contours', f'contour_roi_{timestamp}_{frame_name}')
        os.makedirs(os.path.dirname(contour_save_path), exist_ok=True)
        
        result_img, contour, final_score, largest_lumen_mask = processing.predict_contour_and_save(roi_temp_path, contour_save_path)
        
        contour_info = []
        
        if contour is not None:
            # 计算最大直径并生成可视化图片
            max_diameter_value_in_pixel = 0
            max_diameter_image_path = None
            p1_int = [0, 0]
            p2_int = [0, 0]
            
            try:
                # 创建用于最大直径可视化的保存路径
                max_diameter_save_path = os.path.join('viewer/static/results', f'max_diameter_contour_roi_{timestamp}_{frame_name}')
                os.makedirs(os.path.dirname(max_diameter_save_path), exist_ok=True)
                
                # 创建二值掩码用于计算最大直径
                lumen_binary_mask = np.zeros_like(roi_img[:, :, 0])  # 假设roi是彩色图像
                cv2.drawContours(lumen_binary_mask, [contour], -1, 255, cv2.FILLED)
                
                # 计算最大直径
                max_diameter_value_in_pixel, p1_int, p2_int = processing.find_and_visualize_max_diameter_improved(
                    largest_lumen_mask,
                    contour_save_path,
                    max_diameter_save_path
                )
                
                # 设置返回的图片路径
                max_diameter_image_path = f'/static/results/max_diameter_contour_roi_{timestamp}_{frame_name}'
            except Exception as e:
                print(f"计算最大直径时出错: {e}")
                # 即使计算最大直径出错，也要继续处理其他数据
            
            # 保存测量结果为JSON文件（使用时间戳确保唯一性）
            measurement_data = {
                'contour_image': f'/static/contours/contour_roi_{timestamp}_{frame_name}',
                'max_diameter_image_path': max_diameter_image_path,
                'score': float(final_score),
                'max_diameter_in_pixel': float(max_diameter_value_in_pixel),
                'max_diameter_image': max_diameter_image_path,
                'box': {
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2)
                },
                'line_points': {
                    'p1': {
                        'x': int(p1_int[0]),
                        'y': int(p1_int[1])
                    },
                    'p2': {
                        'x': int(p2_int[0]),
                        'y': int(p2_int[1])
                    }
                }
            }
            
            # 保存为JSON文件
            json_save_path = os.path.join('viewer/static/results', f'measurement_roi_{timestamp}_{frame_name.replace(".jpg", "")}.json')
            try:
                with open(json_save_path, 'w') as f:
                    json.dump(measurement_data, f)
            except Exception as e:
                print(f"保存JSON文件时出错: {e}")
            
            contour_info.append(measurement_data)
        
        return jsonify({
            'success': True,
            'contours': contour_info
        })
        
    except Exception as e:
        print(f"框选测量过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'服务器内部错误: {str(e)}'}), 500
    
if __name__ == '__main__':
    app.run(debug=True)