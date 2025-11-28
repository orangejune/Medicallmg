// viewer.js
class MedicalImageViewer {
    constructor() {
        this.currentFile = null;      // 当前文件
        this.currentFrame = 0;        // 当前帧号
        this.isPlaying = false;       // 播放状态
        this.measurementMode = false; // 测量模式开关
        this.clickPoints = [];        // 存储显示点
        this.measurementPoints = [];  // 存储测量点
        this.pixelSpacing = null;     // 像素间距
        this.unit = null;             // 单位
        this.currentBoxes = null; // 保存当前边界框数据
        
        // 初始化响应式支持
        this.initResponsiveSupport();
    }

    // 初始化响应式支持，使得绘制的元素位置大小正确
    initResponsiveSupport() {
        // 使用 ResizeObserver 监听图像大小变化（现代浏览器）
        const imageDisplay = document.getElementById('image-display');
        if (window.ResizeObserver && imageDisplay) {
            this.resizeObserver = new ResizeObserver(entries => {
                for (let entry of entries) {
                    if (entry.target === imageDisplay && this.currentBoxes) {
                        this.drawBoundingBoxes(this.currentBoxes);
                    }
                }
            });
            this.resizeObserver.observe(imageDisplay);
        } else {
            // 回退到监听窗口大小变化（兼容旧浏览器）
            window.addEventListener('resize', () => {
                if (this.currentBoxes) {
                    setTimeout(() => {
                        this.drawBoundingBoxes(this.currentBoxes);
                    }, 100);
                }
            });
        }
    }

    // 加载文件
    loadFile(fileName) {
        this.currentFile = fileName;
        const imageDisplay = document.getElementById('image-display');
        imageDisplay.src = `images/${fileName}.jpg`;
        
        // 更新显示的帧名称
        this.updateCurrentFrameName(`${fileName}.jpg`);
        
        // 更新文件列表高亮
        document.querySelectorAll('.file-item').forEach(item => {
            item.style.backgroundColor = '';
        });
        document.querySelector(`[data-file="${fileName}"]`).style.backgroundColor = '#e0f7ff';
        
        // 清空测量结果
        document.getElementById('measurement-overlay').innerHTML = '';
    }

    // 批量测量
    autoBatchMeasure() {
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

    //更新帧名称显示
    updateCurrentFrameName(frameName) {
        const frameNameElement = document.getElementById('current-frame-name');
        if (frameNameElement) {
            frameNameElement.textContent = frameName || '无';
        }
    }

    // 加载特定帧
    loadFrame(frameName) {
        const imageDisplay = document.getElementById('image-display');
        imageDisplay.src = `/static/images/${frameName}`;
        
        // 更新当前文件状态
        this.currentFile = frameName.replace('.jpg', '');
        
        // 更新显示的帧名称
        this.updateCurrentFrameName(frameName);
        
        // 清空测量结果
        document.getElementById('measurement-overlay').innerHTML = '';
    }
    // 没有测量结果时能恢复原始标题（暂时没用到
    resetCandidateFramesTitle() {
        const rightSidebar = document.querySelector('.right-sidebar');
        const titleElement = rightSidebar.querySelector('h3');
        if (titleElement) {
            titleElement.textContent = '备选帧';
        }
    }
    // 导入文件
    importFile() {
        // 清空文件列表
        const fileList = document.getElementById('file-list');
        fileList.innerHTML = "";
        
        // 清空候选帧显示
        const candidateFrames = document.getElementById('candidate-frames');
        candidateFrames.innerHTML = '';
        
        // 重置图像显示
        const imageDisplay = document.getElementById('image-display');
        imageDisplay.src = '';
        
        // 清空测量覆盖层
        this.clearMeasurements();
        
        // 重置其他状态
        this.currentFile = null;
        this.currentFrame = 0;
        this.currentBoxes = null;
        this.resetCandidateFramesTitle();
        
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
                
                // 更新当前文件状态和显示
                this.currentFile = fileName;
                this.updateCurrentFrameName(fileNameWithExtension);
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
                    
                    // alert('测量完成，已在图像上标注ROI区域，边界图像显示在备选帧区域');
                });
            } else {
                alert('测量失败: ' + data.error);
            }
        } catch (error) {
            console.error('测量过程中发生错误:', error);
            alert('测量过程中发生错误，请查看控制台了解详情');
        }
    }

    // 添加显示边界图像的方法（单帧
    displayContourImages(contours) {
        const candidateFrames = document.getElementById('candidate-frames');
        // 设置内容区域样式
        candidateFrames.style.maxHeight = 'calc(100vh - 150px)';
        candidateFrames.style.overflowY = 'auto';
        candidateFrames.innerHTML = '';
        
        // 更新备选帧标题，显示边界数量
        const rightSidebar = document.querySelector('.right-sidebar');
        const titleElement = rightSidebar.querySelector('h3');
        if (titleElement) {
            if (contours && contours.length > 0) {
                titleElement.textContent = `备选帧 (共 ${contours.length} 个边界)`;
            } else {
                titleElement.textContent = '备选帧';
            }
        }
        
        if (contours && contours.length > 0) {
            // 创建一个容器用于包装所有边界
            const allContoursContainer = document.createElement('div');
            allContoursContainer.style.marginBottom = '20px';
            allContoursContainer.style.border = '1px solid #ddd';
            allContoursContainer.style.borderRadius = '5px';
            allContoursContainer.style.padding = '10px';
            
            // 添加点击事件，点击时在主区域显示当前帧图像
            allContoursContainer.style.cursor = 'pointer';
            // 获取当前显示的帧名称
            const currentFrameName = this.currentFile + '.jpg';
            allContoursContainer.addEventListener('click', () => {
                this.loadFrame(currentFrameName);
            });
            
            const frameTitle = document.createElement('div');
            frameTitle.textContent = `当前帧: ${this.currentFile}`;
            frameTitle.style.fontWeight = 'bold';
            frameTitle.style.marginBottom = '10px';
            frameTitle.style.fontSize = '14px';
            
            allContoursContainer.appendChild(frameTitle);
            
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
                img.src = contour.max_diameter_image_path;
                img.style.width = '100%';
                img.style.height = 'auto';
                img.style.border = '1px solid #eee';

                // 计算实际直径（如果有像素间距信息）
                let diameterInfo = `直径: `;
                if (this.pixelSpacing) {
                    const diameterInMM = contour.max_diameter_in_pixel * this.pixelSpacing;
                    diameterInfo += `${diameterInMM.toFixed(2)} mm `;
                }
                diameterInfo += `(${contour.max_diameter_in_pixel.toFixed(2)} 像素)`
                
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
                allContoursContainer.appendChild(container);
            });
            
            candidateFrames.appendChild(allContoursContainer);
        } else {
            candidateFrames.innerHTML = '<div>未检测到血管边界</div>';
        }
    }
    // 批量测量
    displayAllContourImages(contours) {
        const candidateFrames = document.getElementById('candidate-frames');
        // 设置内容区域样式
        candidateFrames.style.maxHeight = 'calc(100vh - 150px)';
        candidateFrames.style.overflowY = 'auto';
        
        // 更新备选帧标题，显示边界数量
        const rightSidebar = document.querySelector('.right-sidebar');
        const titleElement = rightSidebar.querySelector('h3');
        if (titleElement) {
            if (contours && contours.length > 0) {
                titleElement.textContent = `备选帧 (共 ${contours.length} 个边界)`;
            } else {
                titleElement.textContent = '备选帧';
            }
        }
        
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
                
                // 添加点击事件，点击时在主区域显示对应的完整帧图像
                frameContainer.style.cursor = 'pointer';
                frameContainer.addEventListener('click', () => {
                    this.loadFrame(frameName);
                });
                
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
                    img.src = contour.max_diameter_image_path;
                    img.style.width = '100%';
                    img.style.height = 'auto';
                    img.style.border = '1px solid #eee';
                    
                    // 计算实际直径（如果有像素间距信息）
                    let diameterInfo = `直径: `;
                    if (this.pixelSpacing) {
                        const diameterInMM = contour.max_diameter_in_pixel * this.pixelSpacing;
                        diameterInfo += `${diameterInMM.toFixed(2)} mm `;
                    }
                    diameterInfo += `(${contour.max_diameter_in_pixel.toFixed(2)} 像素)`
                    
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
                resultText += `边界 ${index + 1}: 直径 ${contour.max_diameter_in_pixel.toFixed(2)} 像素`;
                if (this.pixelSpacing) {
                    const diameterInMM = contour.max_diameter_in_pixel * this.pixelSpacing;
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

    // 绘制边界框（使用坐标映射方法）
    drawBoundingBoxes(boxes) {
        // 保存当前边界框数据以便重新绘制
        this.currentBoxes = boxes;
        
        const overlay = document.getElementById('measurement-overlay');
        overlay.innerHTML = '';
        
        const imageDisplay = document.getElementById('image-display');
        
        if (!imageDisplay.complete || imageDisplay.naturalWidth === 0) {
            console.warn('图像尚未加载完成，无法准确绘制边界框');
            return;
        }
        
        // 获取图像容器的相关信息
        const container = imageDisplay.parentElement;
        const containerRect = container.getBoundingClientRect();
        const imageRect = imageDisplay.getBoundingClientRect();
        
        // 计算图像在容器中的偏移量
        const offsetX = imageRect.left - containerRect.left;
        const offsetY = imageRect.top - containerRect.top;
        
        const naturalWidth = imageDisplay.naturalWidth;
        const naturalHeight = imageDisplay.naturalHeight;
        const displayedWidth = imageRect.width;
        const displayedHeight = imageRect.height;
        
        // 计算缩放比例
        const scaleX = displayedWidth / naturalWidth;
        const scaleY = displayedHeight / naturalHeight;
        
        boxes.forEach(box => {
            // 将原始坐标转换为相对于容器的坐标
            const x1 = box.x1 * scaleX + offsetX;
            const y1 = box.y1 * scaleY + offsetY;
            const x2 = box.x2 * scaleX + offsetX;
            const y2 = box.y2 * scaleY + offsetY;
            console.log(`原图边界框: (${box.x1}, ${box.y1}) - (${box.x2}, ${box.y2})`);
            console.log(`显示边界框: (${x1}, ${y1}) - (${x2}, ${y2})`);
            
            // 创建边界框元素
            const boxElement = document.createElement('div');
            boxElement.style.position = 'absolute';
            boxElement.style.left = `${x1}px`;
            boxElement.style.top = `${y1}px`;
            boxElement.style.width = `${x2 - x1}px`;
            boxElement.style.height = `${y2 - y1}px`;
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

    //--------------------------------手动测量功能------------------------------------
    // 开始测量模式
    startManualMeasurement() {
        this.measurementMode = true;
        this.measurementPoints = [];
        this.clickPoints = []
        
        // // 清除之前的测量标记
        // this.clearMeasurements();
        
        // 添加提示信息
        alert('已进入测量模式，请在图像上点击两个点进行测量');
    }

    // 处理图像点击事件
    handleImageClick(event) {
        if (!this.measurementMode) return;
        
        const imageDisplay = document.getElementById('image-display');
        const overlay = document.getElementById('measurement-overlay');
        
        // 获取图像和覆盖层的位置信息
        const imageRect = imageDisplay.getBoundingClientRect();
        const overlayRect = overlay.getBoundingClientRect();
        
        // 计算点击位置相对于覆盖层的坐标
        const x = event.clientX - overlayRect.left;
        const y = event.clientY - overlayRect.top;
        
        // 计算图像在覆盖层中的位置
        const imageX = imageRect.left - overlayRect.left;
        const imageY = imageRect.top - overlayRect.top;
        
        // 调整坐标为相对于图像的坐标
        const relativeX = x - imageX;
        const relativeY = y - imageY;
        
        // 确保点击位置在图像范围内
        if (relativeX < 0 || relativeX > imageRect.width || relativeY < 0 || relativeY > imageRect.height) {
            console.log('点击位置超出图像范围');
            return;
        }
        
        console.log(`点击位置相对于覆盖层的坐标: (${x.toFixed(2)}, ${y.toFixed(2)})`);
        console.log(`图像在覆盖层中位置: (${imageX.toFixed(2)}, ${imageY.toFixed(2)})`);
        console.log(`点击位置相对于图像坐标: (${relativeX.toFixed(2)}, ${relativeY.toFixed(2)})`);
        console.log(`图像尺寸: ${imageRect.width} x ${imageRect.height}`);
        
        // 保存点击点（相对于图像的坐标）
        this.measurementPoints.push({ 
            x: relativeX, 
            y: relativeY 
        });

        // 保存点击点（显示坐标）
        this.clickPoints.push({ 
            x: x, 
            y: y 
        });

        // 在图像上标记点击点（使用相对于覆盖层的坐标）
        this.markPoint(x, y, this.measurementPoints.length);
        
        // 当有两个点时，计算距离
        if (this.measurementPoints.length === 2) {
            this.calculateAndDisplayDistance();
            // 退出测量模式
            this.measurementMode = false;
        }
    }

    // 标记点击点
    markPoint(x, y, pointNumber) {
        const overlay = document.getElementById('measurement-overlay');
        
        // 创建点标记元素（直接使用相对于覆盖层的坐标）
        const pointElement = document.createElement('div');
        pointElement.style.position = 'absolute';
        pointElement.style.left = `${x - 3}px`;  // 减去半径使点居中
        pointElement.style.top = `${y - 3}px`;   // 减去半径使点居中
        pointElement.style.width = '6px';
        pointElement.style.height = '6px';
        pointElement.style.backgroundColor = 'red';
        pointElement.style.borderRadius = '50%';
        pointElement.style.zIndex = '20';
        pointElement.style.pointerEvents = 'none';
        
        // 添加点编号
        const label = document.createElement('div');
        label.style.position = 'absolute';
        label.style.left = '10px';
        label.style.top = '-20px';
        label.style.color = 'white';
        label.style.backgroundColor = 'rgba(0,0,0,0.7)';
        label.style.padding = '2px 4px';
        label.style.fontSize = '12px';
        label.textContent = pointNumber;
        
        pointElement.appendChild(label);
        overlay.appendChild(pointElement);
        
        console.log(`标记点 ${pointNumber}: 鼠标点击的位置 (${x.toFixed(2)}, ${y.toFixed(2)})`);
    }
    // 计算并显示距离
    calculateAndDisplayDistance() {
        if (this.measurementPoints.length < 2) return;
        
        const point1 = this.measurementPoints[0];
        const point2 = this.measurementPoints[1];
        
        console.log(`计算距离: 点1(${point1.x.toFixed(2)}, ${point1.y.toFixed(2)}) 到 点2(${point2.x.toFixed(2)}, ${point2.y.toFixed(2)})`);
        
        // 获取图像相关信息用于坐标映射
        const imageDisplay = document.getElementById('image-display');
        const naturalWidth = imageDisplay.naturalWidth;
        const naturalHeight = imageDisplay.naturalHeight;
        const displayedWidth = imageDisplay.clientWidth;
        const displayedHeight = imageDisplay.clientHeight;
        
        console.log(`图像信息: 原始尺寸(${naturalWidth}, ${naturalHeight}) 显示尺寸(${displayedWidth}, ${displayedHeight})`);
        
        // 计算坐标缩放比例
        const scaleX = naturalWidth / displayedWidth;
        const scaleY = naturalHeight / displayedHeight;
        
        // 将显示坐标转换为原始图像坐标
        const originalX1 = point1.x * scaleX;
        const originalY1 = point1.y * scaleY;
        const originalX2 = point2.x * scaleX;
        const originalY2 = point2.y * scaleY;
        
        console.log(`对应原始图像坐标: 点1(${originalX1.toFixed(2)}, ${originalY1.toFixed(2)}) 点2(${originalX2.toFixed(2)}, ${originalY2.toFixed(2)})`);
        
        // 计算像素距离
        const pixelDistance = Math.sqrt(
            Math.pow(originalX2 - originalX1, 2) + 
            Math.pow(originalY2 - originalY1, 2)
        );
        
        // 计算实际距离（如果有像素间距信息）
        let realDistance = null;
        if (this.pixelSpacing) {
            realDistance = pixelDistance * this.pixelSpacing;
        }
        
        // 在图像上绘制连线
        this.drawMeasurementLine();
        
        // 显示测量结果
        this.displayManualMeasurementResult(pixelDistance, realDistance);
    }

    // 绘制测量线
    drawMeasurementLine() {
        const point1 = this.clickPoints[0];
        const point2 = this.clickPoints[1];
        const overlay = document.getElementById('measurement-overlay');
        
        // 创建 SVG 元素来绘制线段
        const svgNS = "http://www.w3.org/2000/svg";
        const svg = document.createElementNS(svgNS, "svg");
        
        // 设置 SVG 容器大小和位置
        const minX = Math.min(point1.x, point2.x);
        const minY = Math.min(point1.y, point2.y);
        const maxX = Math.max(point1.x, point2.x);
        const maxY = Math.max(point1.y, point2.y);
        
        svg.style.position = 'absolute';
        svg.style.left = `${minX - 2}px`;
        svg.style.top = `${minY - 2}px`;
        svg.setAttribute('width', maxX - minX + 4);
        svg.setAttribute('height', maxY - minY + 4);
        svg.style.zIndex = '15';
        svg.style.pointerEvents = 'none';
        
        // 创建线段元素
        const line = document.createElementNS(svgNS, "line");
        line.setAttribute('x1', point1.x - minX + 2);
        line.setAttribute('y1', point1.y - minY + 2);
        line.setAttribute('x2', point2.x - minX + 2);
        line.setAttribute('y2', point2.y - minY + 2);
        line.setAttribute('stroke', 'red');
        line.setAttribute('stroke-width', '2');
        
        svg.appendChild(line);
        overlay.appendChild(svg);
    }

    // 显示手动测量结果
    displayManualMeasurementResult(pixelDistance, realDistance) {
        const overlay = document.getElementById('measurement-overlay');
        
        // 查找距离标签并更新文本
        const lineElement = overlay.lastChild;
        if (lineElement && lineElement.children.length > 0) {
            const distanceLabel = lineElement.firstChild;
            let text = `距离: ${pixelDistance.toFixed(2)} 像素`;
            if (realDistance !== null) {
                text += ` (${realDistance.toFixed(2)} mm)`;
            }
            distanceLabel.textContent = text;
        }
        
        // 显示完整结果在控制台
        console.log(`测量结果:`);
        console.log(`像素距离: ${pixelDistance.toFixed(2)} pixels`);
        if (realDistance !== null) {
            console.log(`实际距离: ${realDistance.toFixed(2)} mm`);
        }
        
        // 将测量结果显示在像素位置信息栏
        const pixelSpacingInfo = document.getElementById('pixel-spacing-info');
        if (pixelSpacingInfo) {
            // 清除之前的手动测量结果
            const existingResult = pixelSpacingInfo.querySelector('.manual-measurement-result');
            if (existingResult) {
                existingResult.remove();
            }
            
            // 创建新的测量结果显示元素
            const resultElement = document.createElement('span');
            resultElement.className = 'manual-measurement-result';
            resultElement.style.marginLeft = '20px';
            resultElement.style.color = '#1976d2';
            resultElement.style.fontWeight = 'bold';
            
            if (realDistance !== null) {
                resultElement.textContent = `手动测量结果：${realDistance.toFixed(2)} mm (${pixelDistance.toFixed(2)} 像素)`;
            } else {
                resultElement.textContent = `手动测量结果：${pixelDistance.toFixed(2)} 像素`;
            }
            
            pixelSpacingInfo.appendChild(resultElement);
        }
        
        // 显示在页面上的弹窗通知
        alert(`测量完成:\n像素距离: ${pixelDistance.toFixed(2)} 像素${realDistance !== null ? `\n实际距离: ${realDistance.toFixed(2)} mm` : ''}`);
    }
}

// 初始化查看器
const viewer = new MedicalImageViewer();

// 添加图像点击事件监听器
const imageDisplay = document.getElementById('image-display');
imageDisplay.addEventListener('click', (event) => {
    viewer.handleImageClick(event);
});

// 绑定事件
// 文件列表点击文件名加载对应文件
document.querySelectorAll('.file-item').forEach(item => {
    item.addEventListener('click', function() {
        const fileName = this.getAttribute('data-file');
        viewer.loadFile(fileName);
    });
});

// 绑定按钮事件
document.getElementById('import-btn').onclick = () => viewer.importFile();//导入文件
document.getElementById('info-btn').onclick = () => viewer.showInfo();
document.getElementById('single-measure-btn').onclick = () => viewer.startMeasurement();//单帧测量
document.getElementById('batch-measure-btn').onclick = () => viewer.autoBatchMeasure();//批量测量
document.getElementById('clear-btn').onclick = () => viewer.clearMeasurements();
document.getElementById('export-btn').onclick = () => viewer.exportReport();
document.getElementById('help-btn').onclick = () => viewer.showHelp();
document.getElementById('sort-btn').onclick = () => viewer.sortByScore();
document.getElementById('manual-measure-btn').onclick = () => viewer.startManualMeasurement();

// // 控制按钮事件绑定
// document.getElementById('prev-frame-btn').onclick = () => viewer.prevFrame();
// document.getElementById('play-pause-btn').onclick = () => viewer.playPause();
// document.getElementById('next-frame-btn').onclick = () => viewer.nextFrame();
// document.getElementById('start-measure-btn').onclick = () => viewer.manualMeasure();

// 缩放控制
document.getElementById('zoom-control').addEventListener('input', function() {
    const zoomValue = this.value;
    document.getElementById('zoom-value').textContent = `${zoomValue}%`;
});