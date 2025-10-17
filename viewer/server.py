from flask import Flask, request, jsonify, render_template
import os
import subprocess
import json
from datetime import datetime
import sys
import importlib.util

# 添加当前项目路径到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process-dicom', methods=['POST'])
def process_dicom():
    try:
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


if __name__ == '__main__':
    app.run(debug=True)