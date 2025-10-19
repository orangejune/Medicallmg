// viewer.js
class MedicalImageViewer {
    constructor() {
        this.currentFile = null;      // 当前文件
        this.currentFrame = 0;        // 当前帧号
        this.isPlaying = false;       // 播放状态
        this.measurementMode = false; // 测量模式开关
        this.pixelSpacing = null;     // 像素间距
        this.unit = null;             // 单位
    }

    // 加载文件
    loadFile(fileName) {
        this.currentFile = fileName;
        const imageDisplay = document.getElementById('image-display');
        imageDisplay.src = `images/${fileName}.jpg`;
        
        // 更新文件列表高亮
        document.querySelectorAll('.file-item').forEach(item => {
            item.style.backgroundColor = '';
        });
        document.querySelector(`[data-file="${fileName}"]`).style.backgroundColor = '#e0f7ff';
        
        // 清空测量结果
        document.getElementById('measurement-overlay').innerHTML = '';
    }

    // 手动测量
    manualMeasure() {
        const overlay = document.getElementById('measurement-overlay');
        if (this.measurementMode) {
            overlay.innerHTML = '';
            this.measurementMode = false;
        } else {
            // 执行批量测量
            this.batchMeasure();
            this.measurementMode = true;
        }
    }

    async batchMeasure() {
    try {
        // 获取所有帧文件名
        const fileList = document.getElementById('file-list');
        const frameItems = fileList.querySelectorAll('.file-item');
        const frameNames = Array.from(frameItems).map(item => item.getAttribute('data-file') + '.jpg');
        
        if (frameNames.length === 0) {
            alert('没有可测量的帧');
            return;
        }
        
        // 发送批量测量请求
        const response = await fetch('/batch-measure', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                frame_names: frameNames
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // 在备选帧区域显示所有边界图像和测量结果
            this.displayAllContourImages(data.contours);
            alert(`批量测量完成，共处理了 ${data.contours.length} 个边界`);
        } else {
            alert('批量测量失败: ' + data.error);
        }
    } catch (error) {
        console.error('批量测量过程中发生错误:', error);
        alert('批量测量过程中发生错误，请查看控制台了解详情');
    }
}
    // 播放/暂停
    playPause() {
        this.isPlaying = !this.isPlaying;
        if (this.isPlaying) {
            this.playFrames();
        }
    }

    // 播放帧序列
    playFrames() {
        if (!this.isPlaying) return;
        
        // 这里可以实现自动播放逻辑
        setTimeout(() => {
            this.nextFrame();
            this.playFrames();
        }, 1000);
    }

    // 上一帧
    prevFrame() {
        if (this.currentFrame > 0) {
            this.currentFrame--;
            this.loadFrame(this.currentFrame);
        }
    }

    // 下一帧
    nextFrame() {
        // 这里可以根据实际帧数调整
        this.currentFrame++;
        this.loadFrame(this.currentFrame);
    }

    // 加载特定帧
    loadFrame(frameName) {
        const imageDisplay = document.getElementById('image-display');
        imageDisplay.src = `/static/images/${frameName}`;
        
        // 更新当前文件状态
        this.currentFile = frameName.replace('.jpg', '');
        
        // 清空测量结果
        document.getElementById('measurement-overlay').innerHTML = '';
    }

    // 导入文件
    importFile() {
        // 实现文件导入逻辑
        const input = document.createElement('input');
        input.type = 'file';
        // input.accept = '.dcm,.dicom,.jpg,.png';
        input.onchange = (e) => {
            const file = e.target.files[0];
            if (file) {
                // 处理文件上传
                console.log('文件上传:', file.name);
                const formData = new FormData();
                formData.append('file', file);

                fetch('/process-dicom', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        // 保存像素距离信息
                        this.pixelSpacing = data.pixel_spacing;
                        this.unit = data.unit;
                        
                        // 显示像素距离信息
                        this.displayPixelSpacing();
                        
                        // 清空旧文件列表
                        const fileList = document.getElementById('file-list');
                        fileList.innerHTML = "";

                        // 动态添加新文件项
                        data.frames.forEach(frameName => {
                            const li = document.createElement('li');
                            li.className = 'file-item';
                            li.setAttribute('data-file', frameName.split('.')[0]); // 提取文件名
                            li.textContent = frameName;
                            li.onclick = () => this.loadFrame(frameName);
                            fileList.appendChild(li);
                        });

                        // 自动加载第一帧
                        if (data.frames.length > 0) {
                            this.loadFrame(data.frames[0]);
                        }

                        alert('DICOM 转换成功！已加载帧列表。');
                    } else {
                        alert('转换失败：' + data.error);
                    }
                })
                .catch(err => {
                    console.error(err);
                    alert('上传失败，请重试。');
                });
            }
        };
        input.click();
    }

    // 显示像素距离信息
    displayPixelSpacing() {
        const pixelSpacingElement = document.getElementById('pixel-spacing-value');
        if (this.pixelSpacing && this.unit) {
            pixelSpacingElement.textContent = `${this.pixelSpacing.toFixed(4)} mm/pixel`;
        } else {
            pixelSpacingElement.textContent = '无法获取像素距离信息';
        }
    }

    // 显示信息
    showInfo() {
        // 实现信息视图逻辑
        let info = '显示患者信息和影像详情\n';
        if (this.pixelSpacing && this.unit) {
            info += `像素距离: ${this.pixelSpacing.toFixed(4)} ${this.unit}\n`;
        } else {
            info += '像素距离: 未获取\n';
        }
        alert(info);
    }

    // 导出报告
    exportReport() {
        // 实现报告导出逻辑
        alert('导出测量结果报告');
    }

    // 排序功能
    sortByScore() {
        // 实现按打分质量排序逻辑
        alert('按打分质量排序');
    }

    // 帮助功能
    showHelp() {
        // 实现帮助功能
        alert('帮助文档');
    }

    // 开始测量
    onImageLoaded(callback) {
        const imageDisplay = document.getElementById('image-display');
        if (imageDisplay.complete && imageDisplay.naturalWidth !== 0) {
            // 图像已经加载完成
            callback();
        } else {
            // 等待图像加载完成
            imageDisplay.onload = callback;
        }
    }

    // 修改 startMeasurement 方法以确保图像已加载
    async startMeasurement() {
        // 检查是否有当前文件，如果没有则尝试从显示的图像中获取
        let fileName = this.currentFile;
        
        if (!fileName) {
            // 尝试从当前显示的图像中获取文件名
            const imageDisplay = document.getElementById('image-display');
            if (imageDisplay && imageDisplay.src) {
                const src = imageDisplay.src;
                const urlParts = src.split('/');
                const fileNameWithExtension = urlParts[urlParts.length - 1];
                fileName = fileNameWithExtension.replace('.jpg', '');
            }
        }
        
        if (!fileName) {
            alert('请先选择一个文件');
            return;
        }
        
        try {
            // 发送请求到后端进行测量
            const response = await fetch('/measure-frame', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    frame_name: fileName + '.jpg'
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // 确保图像已加载后再绘制边界框
                this.onImageLoaded(() => {
                    // 在图像上绘制边界框
                    this.drawBoundingBoxes(data.boxes);
                    
                    // 在备选帧区域显示边界图像
                    this.displayContourImages(data.contours);
                    
                    // 显示测量结果
                    this.displayMeasurementResults(data.contours);
                    
                    alert('测量完成，已在图像上标注ROI区域，边界图像显示在备选帧区域');
                });
            } else {
                alert('测量失败: ' + data.error);
            }
        } catch (error) {
            console.error('测量过程中发生错误:', error);
            alert('测量过程中发生错误，请查看控制台了解详情');
        }
    }

    // 添加显示边界图像的方法
    displayContourImages(contours) {
        const candidateFrames = document.getElementById('candidate-frames');
        candidateFrames.innerHTML = '';
        
        if (contours && contours.length > 0) {
            contours.forEach((contour, index) => {
                const container = document.createElement('div');
                container.style.marginBottom = '15px';
                container.style.border = '1px solid #ccc';
                container.style.padding = '10px';
                
                const title = document.createElement('div');
                title.textContent = `边界 ${index + 1}`;
                title.style.fontWeight = 'bold';
                title.style.marginBottom = '5px';
                
                const img = document.createElement('img');
                img.src = contour.contour_image;
                img.style.width = '100%';
                img.style.height = 'auto';
                
                const info = document.createElement('div');
                info.innerHTML = `
                    <div>置信度: ${contour.confidence.toFixed(2)}</div>
                    <div>直径: ${contour.diameter.toFixed(2)} 像素</div>
                `;
                info.style.fontSize = '12px';
                info.style.marginTop = '5px';
                
                container.appendChild(title);
                container.appendChild(img);
                container.appendChild(info);
                candidateFrames.appendChild(container);
            });
        } else {
            candidateFrames.innerHTML = '<div>未检测到血管边界</div>';
        }
    }
    displayAllContourImages(contours) {
        const candidateFrames = document.getElementById('candidate-frames');
        candidateFrames.innerHTML = '';
        
        if (contours && contours.length > 0) {
            // 按帧分组
            const groupedContours = {};
            contours.forEach(contour => {
                if (!groupedContours[contour.frame_name]) {
                    groupedContours[contour.frame_name] = [];
                }
                groupedContours[contour.frame_name].push(contour);
            });
            
            // 显示每帧的结果
            Object.keys(groupedContours).forEach(frameName => {
                const frameContainer = document.createElement('div');
                frameContainer.style.marginBottom = '20px';
                frameContainer.style.border = '1px solid #ddd';
                frameContainer.style.borderRadius = '5px';
                frameContainer.style.padding = '10px';
                
                const frameTitle = document.createElement('div');
                frameTitle.textContent = `帧: ${frameName.replace('.jpg', '')}`;
                frameTitle.style.fontWeight = 'bold';
                frameTitle.style.marginBottom = '10px';
                frameTitle.style.fontSize = '14px';
                
                frameContainer.appendChild(frameTitle);
                
                // 显示该帧的所有边界
                groupedContours[frameName].forEach((contour, index) => {
                    const container = document.createElement('div');
                    container.style.marginBottom = '15px';
                    container.style.border = '1px solid #ccc';
                    container.style.padding = '8px';
                    container.style.backgroundColor = '#f9f9f9';
                    
                    const title = document.createElement('div');
                    title.textContent = `血管 ${index + 1}`;
                    title.style.fontWeight = 'bold';
                    title.style.marginBottom = '5px';
                    title.style.fontSize = '12px';
                    
                    const img = document.createElement('img');
                    img.src = contour.contour_image;
                    img.style.width = '100%';
                    img.style.height = 'auto';
                    img.style.border = '1px solid #eee';
                    
                    // 计算实际直径（如果有像素间距信息）
                    let diameterInfo = `直径: ${contour.diameter.toFixed(2)} 像素`;
                    if (this.pixelSpacing) {
                        const diameterInMM = contour.diameter * this.pixelSpacing;
                        diameterInfo += ` (${diameterInMM.toFixed(2)} mm)`;
                    }
                    
                    const info = document.createElement('div');
                    info.innerHTML = `
                        <div style="font-size: 11px; margin-top: 5px;">
                            <div>置信度: ${contour.confidence.toFixed(2)}</div>
                            <div>${diameterInfo}</div>
                        </div>
                    `;
                    
                    container.appendChild(title);
                    container.appendChild(img);
                    container.appendChild(info);
                    frameContainer.appendChild(container);
                });
                
                candidateFrames.appendChild(frameContainer);
            });
        } else {
            candidateFrames.innerHTML = '<div style="text-align: center; padding: 20px; color: #666;">未检测到血管边界</div>';
        }
    }
    // 添加显示测量结果的方法
    displayMeasurementResults(contours) {
        // 如果有像素间距信息，可以计算实际尺寸
        let resultText = '测量结果:\n';
        
        if (contours && contours.length > 0) {
            contours.forEach((contour, index) => {
                resultText += `边界 ${index + 1}: 直径 ${contour.diameter.toFixed(2)} 像素`;
                if (this.pixelSpacing) {
                    const diameterInMM = contour.diameter * this.pixelSpacing;
                    resultText += ` (${diameterInMM.toFixed(2)} mm)`;
                }
                resultText += '\n';
            });
        } else {
            resultText += '未检测到血管边界';
        }
        
        // 可以将结果显示在像素距离信息旁边或其他位置
        console.log(resultText);
    }

    // 绘制边界框
    drawBoundingBoxes(boxes) {
        const overlay = document.getElementById('measurement-overlay');
        overlay.innerHTML = '';
        
        // 获取显示图像的元素和尺寸
        const imageDisplay = document.getElementById('image-display');
        const displayedWidth = imageDisplay.offsetWidth;
        const displayedHeight = imageDisplay.offsetHeight;
        
        // 获取图像的自然尺寸（原始尺寸）
        const naturalWidth = imageDisplay.naturalWidth;
        const naturalHeight = imageDisplay.naturalHeight;
        
        // 计算缩放比例
        const scaleX = displayedWidth / naturalWidth;
        const scaleY = displayedHeight / naturalHeight;
        
        boxes.forEach(box => {
            // 将原始坐标转换为显示坐标
            const displayX1 = box.x1 * scaleX;
            const displayY1 = box.y1 * scaleY;
            const displayX2 = box.x2 * scaleX;
            const displayY2 = box.y2 * scaleY;
            
            // 创建边界框元素
            const boxElement = document.createElement('div');
            boxElement.style.position = 'absolute';
            boxElement.style.left = `${displayX1}px`;
            boxElement.style.top = `${displayY1}px`;
            boxElement.style.width = `${displayX2 - displayX1}px`;
            boxElement.style.height = `${displayY2 - displayY1}px`;
            boxElement.style.border = '2px solid yellow';
            boxElement.style.boxSizing = 'border-box';
            boxElement.style.pointerEvents = 'none';
            boxElement.style.zIndex = '10';
            
            // 添加置信度标签
            const label = document.createElement('div');
            label.style.position = 'absolute';
            label.style.top = '-20px';
            label.style.left = '0';
            label.style.background = 'rgba(255, 255, 0, 0.7)';
            label.style.color = 'black';
            label.style.padding = '2px 4px';
            label.style.fontSize = '12px';
            label.style.whiteSpace = 'nowrap';
            label.textContent = `Class: ${box.class}, Conf: ${box.confidence.toFixed(2)}`;
            
            boxElement.appendChild(label);
            overlay.appendChild(boxElement);
        });
    }
    
    // 清除测量标记
    clearMeasurements() {
        const overlay = document.getElementById('measurement-overlay');
        overlay.innerHTML = '';
    }
}

// 初始化查看器
const viewer = new MedicalImageViewer();

// 绑定事件
// 文件列表点击文件名加载对应文件
document.querySelectorAll('.file-item').forEach(item => {
    item.addEventListener('click', function() {
        const fileName = this.getAttribute('data-file');
        viewer.loadFile(fileName);
    });
});

// 绑定按钮事件
document.querySelector('.menu-button:nth-child(1)').onclick = () => viewer.importFile();
document.querySelector('.menu-button:nth-child(2)').onclick = () => viewer.showInfo();
document.querySelector('.menu-button:nth-child(3)').onclick = () => viewer.startMeasurement();
document.querySelector('.menu-button:nth-child(4)').onclick = () => viewer.manualMeasure(); //批量测量
document.querySelector('.menu-button:nth-child(5)').onclick = () => viewer.clearMeasurements();
document.querySelector('.menu-button:nth-child(6)').onclick = () => viewer.exportReport();
document.querySelector('.menu-button:nth-child(7)').onclick = () => viewer.showHelp();
document.querySelector('.menu-button:nth-child(8)').onclick = () => viewer.sortByScore();

// 控制按钮事件
document.querySelector('.menu-button:nth-child(1)').onclick = () => viewer.prevFrame();
document.querySelector('.menu-button:nth-child(2)').onclick = () => viewer.playPause();
document.querySelector('.menu-button:nth-child(3)').onclick = () => viewer.nextFrame();
document.querySelector('.menu-button:nth-child(4)').onclick = () => viewer.manualMeasure();

// 缩放控制
document.getElementById('zoom-control').addEventListener('input', function() {
    const zoomValue = this.value;
    document.getElementById('zoom-value').textContent = `${zoomValue}%`;
});