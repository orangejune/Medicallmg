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
            
            // 在开始批量测量前清空右侧结果和中间图像的标记
            const candidateFrames = document.getElementById('candidate-frames');
            candidateFrames.innerHTML = '';
            this.clearMeasurements();
            
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
                // alert(`批量测量完成，共处理了 ${data.contours.length} 个边界`);
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
        this.clearMeasurements();
        
        // 加载并显示测量线（如果存在）
        setTimeout(() => {
            this.loadAndDisplayMeasurementLine(frameName);
        }, 100); // 延迟执行，确保图像开始加载
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

                        // alert('DICOM 转换成功！已加载帧列表。');
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
        
        // 在开始测量前清空右侧结果和中间图像的标记
        const candidateFrames = document.getElementById('candidate-frames');
        candidateFrames.innerHTML = '';
        this.clearMeasurements();
        
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
                    
                    // 在图像上绘制轮廓
                    if (data.contours && data.contours.length > 0) {
                        // 绘制第一个轮廓作为示例
                        this.drawContourFromData(data.contours[0]);
                    }
                    
                    // 在备选帧区域显示边界图像
                    this.displayContourImages(data.contours);
                    
                    // 显示测量结果
                    this.displayMeasurementResults(data.contours);
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
                this.clearMeasurements(); // 点击时先清除标记
                this.loadFrame(currentFrameName);
            });
            
            const frameTitle = document.createElement('div');
            frameTitle.textContent = `当前帧: ${this.currentFile}`;
            frameTitle.style.fontWeight = 'bold';
            frameTitle.style.marginBottom = '10px';
            frameTitle.style.fontSize = '14px';
            
            allContoursContainer.appendChild(frameTitle);
            
            // 按评分从高到低排序
            contours.sort((a, b) => b.score - a.score);
            
            contours.forEach((contour, index) => {
                const container = document.createElement('div');
                container.style.marginBottom = '15px';
                container.style.border = '1px solid #ccc';
                container.style.padding = '10px';
                
                const title = document.createElement('div');
                title.textContent = `边界 ${index + 1} (评分: ${contour.score.toFixed(3)})`;
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
    // 批量测量,按评分排序
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
            // 按照评分从高到低排序
            contours.sort((a, b) => b.score - a.score);
            
            // 按帧分组
            const groupedContours = {};
            contours.forEach(contour => {
                const frameName = contour.frame_name || 'unknown.jpg';
                if (!groupedContours[frameName]) {
                    groupedContours[frameName] = [];
                }
                groupedContours[frameName].push(contour);
            });
            
            // 显示每帧的结果
            Object.keys(groupedContours).forEach(frameName => {
                const frameContainer = document.createElement('div');
                frameContainer.style.marginBottom = '20px';
                frameContainer.style.border = '1px solid #ddd';
                frameContainer.style.borderRadius = '5px';
                frameContainer.style.padding = '10px';
                
                // 添加点击事件，点击时在主区域显示对应的完整帧图像和测量结果
                frameContainer.style.cursor = 'pointer';
                frameContainer.addEventListener('click', () => {
                    this.clearMeasurements(); // 点击时先清除标记
                    this.loadFrame(frameName);
                    
                    // 在主图像上绘制该帧的第一个轮廓和测量线（如果有）
                    setTimeout(() => {
                        const frameContours = groupedContours[frameName];
                        if (frameContours && frameContours.length > 0) {
                            const firstContour = frameContours[0];
                            
                            // 绘制测量线
                            if (firstContour.line_points && firstContour.box) {
                                this.drawMeasurementLineFromData(firstContour);
                            }
                            
                            // 绘制轮廓
                            if (firstContour.contour_points && firstContour.contour_points.length > 0) {
                                this.drawContourFromData(firstContour);
                            }
                        }
                    }, 100);
                });
                
                const frameTitle = document.createElement('div');
                frameTitle.textContent = `帧: ${frameName.replace('.jpg', '')}`;
                frameTitle.style.fontWeight = 'bold';
                frameTitle.style.marginBottom = '10px';
                frameTitle.style.fontSize = '14px';
                
                frameContainer.appendChild(frameTitle);
                
                // 显示该帧的所有边界，按评分排序
                groupedContours[frameName].sort((a, b) => b.score - a.score);
                groupedContours[frameName].forEach((contour, index) => {
                    const container = document.createElement('div');
                    container.style.marginBottom = '15px';
                    container.style.border = '1px solid #ccc';
                    container.style.padding = '8px';
                    container.style.backgroundColor = '#f9f9f9';
                    
                    const title = document.createElement('div');
                    title.textContent = `血管 ${index + 1} (评分: ${contour.score.toFixed(3)})`;
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
        
        // 清除现有内容
        this.clearMeasurements();
        
        const overlay = document.getElementById('measurement-overlay');
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
        if (overlay) {
            // 移除所有子元素
            while (overlay.firstChild) {
                overlay.removeChild(overlay.firstChild);
            }
        }
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
        // alert(`测量完成:\n像素距离: ${pixelDistance.toFixed(2)} 像素${realDistance !== null ? `\n实际距离: ${realDistance.toFixed(2)} mm` : ''}`);
    }
    // 点击备选帧后中间图像显示测量结果位置标志
    async loadAndDisplayMeasurementLine(frameName) {
        // 先清除现有的标记
        this.clearMeasurements();
        
        try {
            const response = await fetch(`/get-measurement-result?frame_name=${frameName}`);
            const data = await response.json();
            
            if (!data.error) {
                // 等待图像加载完成后再绘制
                this.onImageLoaded(() => {
                    // 绘制测量线
                    this.drawMeasurementLineFromData(data);
                    
                    // 绘制轮廓
                    this.drawContourFromData(data);
                });
            }
        } catch (error) {
            console.log('未找到测量结果或加载失败:', error);
        }
    }
    // 添加加载和显示测量线的方法
    drawMeasurementLineFromData(measurementData) {
        const overlay = document.getElementById('measurement-overlay');
        const imageDisplay = document.getElementById('image-display');
        
        if (!imageDisplay.complete || imageDisplay.naturalWidth === 0) {
            console.warn('图像尚未加载完成，无法准确绘制测量线');
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
        
        // 检查是否存在测量线信息
        if (measurementData.line_points && measurementData.box) {
            // 获取ROI边界框在原始图像中的位置
            const roiX1 = measurementData.box.x1;
            const roiY1 = measurementData.box.y1;
            
            // 将ROI内的点坐标转换为全图坐标
            const p1_x = roiX1 + measurementData.line_points.p1.x;
            const p1_y = roiY1 + measurementData.line_points.p1.y;
            const p2_x = roiX1 + measurementData.line_points.p2.x;
            const p2_y = roiY1 + measurementData.line_points.p2.y;
            
            // 将原始坐标转换为相对于容器的坐标
            const x1 = p1_x * scaleX + offsetX;
            const y1 = p1_y * scaleY + offsetY;
            const x2 = p2_x * scaleX + offsetX;
            const y2 = p2_y * scaleY + offsetY;
            
            // 创建测量线元素
            const lineElement = document.createElement('div');
            lineElement.style.position = 'absolute';
            lineElement.style.left = `${Math.min(x1, x2)}px`;
            lineElement.style.top = `${Math.min(y1, y2)}px`;
            lineElement.style.width = `${Math.abs(x2 - x1)}px`;
            lineElement.style.height = `${Math.abs(y2 - y1)}px`;
            lineElement.style.pointerEvents = 'none';
            lineElement.style.zIndex = '11';
            
            // 使用SVG绘制线条
            const svgNS = "http://www.w3.org/2000/svg";
            const svg = document.createElementNS(svgNS, "svg");
            svg.setAttribute('width', '100%');
            svg.setAttribute('height', '100%');
            svg.style.position = 'absolute';
            svg.style.top = '0';
            svg.style.left = '0';
            
            const line = document.createElementNS(svgNS, "line");
            line.setAttribute('x1', x1 < x2 ? 0 : Math.abs(x2 - x1));
            line.setAttribute('y1', y1 < y2 ? 0 : Math.abs(y2 - y1));
            line.setAttribute('x2', x1 < x2 ? Math.abs(x2 - x1) : 0);
            line.setAttribute('y2', y1 < y2 ? Math.abs(y2 - y1) : 0);
            line.setAttribute('stroke', 'rgb(255, 255, 0)'); 
            line.setAttribute('stroke-width', '2');
            
            svg.appendChild(line);
            lineElement.appendChild(svg);
            overlay.appendChild(lineElement);
        }
    }

    //在中间图像绘制半透明边界
    drawContourFromData(contourData) {
        const overlay = document.getElementById('measurement-overlay');
        const imageDisplay = document.getElementById('image-display');
        
        if (!imageDisplay.complete || imageDisplay.naturalWidth === 0) {
            console.warn('图像尚未加载完成，无法准确绘制轮廓');
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
        
        // 检查是否存在轮廓点信息
        if (contourData.contour_points && contourData.contour_points.length > 0 && contourData.box) {
            // 获取ROI边界框在原始图像中的位置
            const roiX1 = contourData.box.x1;
            const roiY1 = contourData.box.y1;
            
            // 验证数据有效性
            if (isNaN(roiX1) || isNaN(roiY1) || isNaN(scaleX) || isNaN(scaleY)) {
                console.error('坐标计算出现 NaN 值');
                return;
            }
            
            // 创建SVG元素用于绘制轮廓
            const svgNS = "http://www.w3.org/2000/svg";
            const svg = document.createElementNS(svgNS, "svg");
            svg.setAttribute('width', '100%');
            svg.setAttribute('height', '100%');
            svg.style.position = 'absolute';
            svg.style.top = '0';
            svg.style.left = '0';
            svg.style.pointerEvents = 'none';
            svg.style.zIndex = '13'; // 略高于测量线
            
            // 创建多边形元素表示轮廓
            const polygon = document.createElementNS(svgNS, "polygon");
            
            try {
                // 转换轮廓点坐标
                const validPoints = [];
                for (let i = 0; i < contourData.contour_points.length; i++) {
                    const point = contourData.contour_points[i];
                    
                    // 处理三层嵌套数组 [[[x, y]], [[x, y]], ...]
                    let xCoord, yCoord;
                    
                    // 检查是否为三层嵌套数组
                    if (Array.isArray(point) && 
                        point.length > 0 && 
                        Array.isArray(point[0]) && 
                        point[0].length >= 2) {
                        // 三层嵌套: [[x, y]]
                        xCoord = point[0][0];
                        yCoord = point[0][1];
                    } else if (Array.isArray(point) && point.length >= 2) {
                        // 两层嵌套: [x, y]
                        xCoord = point[0];
                        yCoord = point[1];
                    } else {
                        console.warn('跳过无效点:', point);
                        continue;
                    }
                    
                    // 检查坐标是否有效
                    if (isNaN(xCoord) || isNaN(yCoord)) {
                        console.warn('跳过无效点:', point);
                        continue;
                    }
                    
                    // 将ROI内的点坐标转换为全图坐标
                    const globalX = roiX1 + xCoord;
                    const globalY = roiY1 + yCoord;
                    
                    // 将原始坐标转换为相对于容器的坐标
                    const displayX = globalX * scaleX + offsetX;
                    const displayY = globalY * scaleY + offsetY;
                    
                    // 检查转换后的坐标是否有效
                    if (isNaN(displayX) || isNaN(displayY)) {
                        console.warn('坐标转换后出现 NaN 值:', {globalX, globalY, scaleX, scaleY, offsetX, offsetY});
                        continue;
                    }
                    
                    validPoints.push(`${displayX},${displayY}`);
                }
                
                // 只有在有足够点的情况下才绘制
                if (validPoints.length >= 3) {
                    const pointsString = validPoints.join(' ');
                    polygon.setAttribute('points', pointsString);
                    polygon.setAttribute('fill', 'none');
                    polygon.setAttribute('stroke', 'rgba(255, 0, 0, 0.7)');
                    polygon.setAttribute('stroke-width', '1.5');
                    
                    svg.appendChild(polygon);
                    overlay.appendChild(svg);
                    console.log('成功绘制轮廓，使用点数:', validPoints.length);
                } else {
                    console.warn('有效点数不足，无法绘制轮廓。有效点数:', validPoints.length);
                }
            } catch (error) {
                console.error('绘制轮廓时出错:', error);
            }
        } else {
            console.log('没有有效的轮廓数据可供绘制');
        }
    }


    //--------------------------------------------ROI框选测量-----------------------------------------
    // 开始框选测量模式
    drawRoiMeasurement() {
        this.measurementMode = 'roi';
        this.isDrawing = false;
        this.startPoint = null;
        
        // 添加提示信息
        alert('已进入框选测量模式，请在图像上拖拽绘制矩形区域');
    }

    // 处理鼠标按下事件
    handleMouseDown(event) {
        if (this.measurementMode !== 'roi') return;
        
        event.preventDefault();
        
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
            return;
        }
        
        this.isDrawing = true;
        this.startPoint = { 
            x: relativeX, 
            y: relativeY,
            displayX: x,
            displayY: y
        };
        
        // 清除之前的框选框
        this.clearROIRectangle();
    }

    // 处理鼠标移动事件
    handleMouseMove(event) {
        if (this.measurementMode !== 'roi' || !this.isDrawing || !this.startPoint) return;
        
        event.preventDefault();
        
        const imageDisplay = document.getElementById('image-display');
        const overlay = document.getElementById('measurement-overlay');
        
        // 获取图像和覆盖层的位置信息
        const imageRect = imageDisplay.getBoundingClientRect();
        const overlayRect = overlay.getBoundingClientRect();
        
        // 计算当前位置相对于覆盖层的坐标
        const x = event.clientX - overlayRect.left;
        const y = event.clientY - overlayRect.top;
        
        // 计算图像在覆盖层中的位置
        const imageX = imageRect.left - overlayRect.left;
        const imageY = imageRect.top - overlayRect.top;
        
        // 调整坐标为相对于图像的坐标
        const relativeX = x - imageX;
        const relativeY = y - imageY;
        
        // 更新临时框选框（使用显示坐标）
        this.updateTempRectangle(this.startPoint.displayX, this.startPoint.displayY, x, y);
    }

    // 处理鼠标释放事件
    handleMouseUp(event) {
        if (this.measurementMode !== 'roi' || !this.isDrawing || !this.startPoint) return;
        
        event.preventDefault();
        
        const imageDisplay = document.getElementById('image-display');
        const overlay = document.getElementById('measurement-overlay');
        
        // 获取图像和覆盖层的位置信息
        const imageRect = imageDisplay.getBoundingClientRect();
        const overlayRect = overlay.getBoundingClientRect();
        const containerRect = imageDisplay.parentElement.getBoundingClientRect();
        
        // 计算当前位置相对于覆盖层的坐标
        const x = event.clientX - overlayRect.left;
        const y = event.clientY - overlayRect.top;
        
        // 计算图像在覆盖层中的位置
        const imageX = imageRect.left - overlayRect.left;
        const imageY = imageRect.top - overlayRect.top;
        
        // 调整坐标为相对于图像的坐标
        const relativeX = x - imageX;
        const relativeY = y - imageY;
        
        // 确定矩形框的坐标（相对于图像的坐标）
        const x1 = Math.min(this.startPoint.x, relativeX);
        const y1 = Math.min(this.startPoint.y, relativeY);
        const x2 = Math.max(this.startPoint.x, relativeX);
        const y2 = Math.max(this.startPoint.y, relativeY);
        
        // 确保框选区域有效
        if (x2 - x1 > 5 && y2 - y1 > 5) {
            // 创建永久框选框（使用显示坐标）
            this.createROIRectangle(this.startPoint.displayX, this.startPoint.displayY, x, y);
            
            // 执行框选测量（使用相对于图像的坐标进行转换）
            this.performROIMeasurement(x1, y1, x2, y2);
        }
        
        this.isDrawing = false;
        this.startPoint = null;
        
        // 退出框选模式
        this.measurementMode = false;
    }

    // 更新临时框选框
    updateTempRectangle(x1, y1, x2, y2) {
        // 清除之前的临时框选框
        const existingTempRect = document.getElementById('temp-roi-rectangle');
        if (existingTempRect) {
            existingTempRect.remove();
        }
        
        const overlay = document.getElementById('measurement-overlay');
        
        const rectElement = document.createElement('div');
        rectElement.id = 'temp-roi-rectangle';
        rectElement.style.position = 'absolute';
        rectElement.style.left = `${Math.min(x1, x2)}px`;
        rectElement.style.top = `${Math.min(y1, y2)}px`;
        rectElement.style.width = `${Math.abs(x2 - x1)}px`;
        rectElement.style.height = `${Math.abs(y2 - y1)}px`;
        rectElement.style.border = '2px dashed blue';
        rectElement.style.backgroundColor = 'rgba(0, 0, 255, 0.2)';
        rectElement.style.pointerEvents = 'none';
        rectElement.style.zIndex = '9';
        
        overlay.appendChild(rectElement);
    }

    // 创建永久框选框
    createROIRectangle(x1, y1, x2, y2) {
        // 清除临时框选框
        const tempRect = document.getElementById('temp-roi-rectangle');
        if (tempRect) {
            tempRect.remove();
        }
        
        const overlay = document.getElementById('measurement-overlay');
        
        const rectElement = document.createElement('div');
        rectElement.id = 'roi-rectangle';
        rectElement.style.position = 'absolute';
        rectElement.style.left = `${Math.min(x1, x2)}px`;
        rectElement.style.top = `${Math.min(y1, y2)}px`;
        rectElement.style.width = `${Math.abs(x2 - x1)}px`;
        rectElement.style.height = `${Math.abs(y2 - y1)}px`;
        rectElement.style.border = '2px solid blue';
        rectElement.style.backgroundColor = 'rgba(0, 0, 255, 0.2)';
        rectElement.style.pointerEvents = 'none';
        rectElement.style.zIndex = '9';
        
        overlay.appendChild(rectElement);
    }

    // 清除框选矩形
    clearROIRectangle() {
        const tempRect = document.getElementById('temp-roi-rectangle');
        if (tempRect) {
            tempRect.remove();
        }
        
        const rect = document.getElementById('roi-rectangle');
        if (rect) {
            rect.remove();
        }
    }

    // 执行框选测量
    async performROIMeasurement(x1, y1, x2, y2) {
        // 检查是否有当前文件
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
        
        // 获取图像相关信息用于坐标映射
        const imageDisplay = document.getElementById('image-display');
        const naturalWidth = imageDisplay.naturalWidth;
        const naturalHeight = imageDisplay.naturalHeight;
        const displayedWidth = imageDisplay.clientWidth;
        const displayedHeight = imageDisplay.clientHeight;
        
        // 计算坐标缩放比例
        const scaleX = naturalWidth / displayedWidth;
        const scaleY = naturalHeight / displayedHeight;
        
        // 将显示坐标转换为原始图像坐标
        const originalX1 = x1 * scaleX;
        const originalY1 = y1 * scaleY;
        const originalX2 = x2 * scaleX;
        const originalY2 = y2 * scaleY;
        
        // 在开始测量前清空右侧结果
        const candidateFrames = document.getElementById('candidate-frames');
        candidateFrames.innerHTML = '';
        
        try {
            // 发送请求到后端进行框选测量
            const response = await fetch('/roi-measure', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    frame_name: fileName + '.jpg',
                    roi: {
                        x1: originalX1,
                        y1: originalY1,
                        x2: originalX2,
                        y2: originalY2
                    }
                })
            });
            
            // 检查响应是否成功
            if (!response.ok) {
                const errorText = await response.text();
                console.error('服务器返回错误:', response.status, errorText);
                alert(`服务器错误 (${response.status}): ${errorText}`);
                return;
            }
            
            // 尝试解析JSON
            const data = await response.json();
            
            if (data.success) {
                // 在备选帧区域显示边界图像
                this.displayROIMeasurementResults(data.contours);
            } else {
                alert('框选测量失败: ' + data.error);
            }
        } catch (error) {
            console.error('框选测量过程中发生错误:', error);
            if (error instanceof SyntaxError) {
                alert('服务器返回了无效的响应格式。请检查服务器日志了解详细信息。');
            } else {
                alert('框选测量过程中发生错误，请查看控制台了解详情');
            }
        }
    }

    // 显示框选测量结果
    displayROIMeasurementResults(contours) {
        const candidateFrames = document.getElementById('candidate-frames');
        // 设置内容区域样式
        candidateFrames.style.maxHeight = 'calc(100vh - 150px)';
        candidateFrames.style.overflowY = 'auto';
        
        // 更新备选帧标题
        const rightSidebar = document.querySelector('.right-sidebar');
        const titleElement = rightSidebar.querySelector('h3');
        if (titleElement) {
            if (contours && contours.length > 0) {
                titleElement.textContent = `框选测量结果 (共 ${contours.length} 个边界)`;
            } else {
                titleElement.textContent = '框选测量结果';
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
                this.clearMeasurements(); // 点击时先清除标记
                this.loadFrame(currentFrameName);
            });
            
            const frameTitle = document.createElement('div');
            frameTitle.textContent = `当前帧: ${this.currentFile} (框选测量)`;
            frameTitle.style.fontWeight = 'bold';
            frameTitle.style.marginBottom = '10px';
            frameTitle.style.fontSize = '14px';
            
            allContoursContainer.appendChild(frameTitle);
            
            // 按评分从高到低排序
            contours.sort((a, b) => b.score - a.score);
            
            contours.forEach((contour, index) => {
                const container = document.createElement('div');
                container.style.marginBottom = '15px';
                container.style.border = '1px solid #ccc';
                container.style.padding = '10px';
                
                const title = document.createElement('div');
                title.textContent = `边界 ${index + 1} (评分: ${contour.score.toFixed(3)})`;
                title.style.fontWeight = 'bold';
                title.style.marginBottom = '5px';
                
                const img = document.createElement('img');
                // 添加时间戳参数避免浏览器缓存
                const timestamp = new Date().getTime();
                img.src = contour.max_diameter_image_path + `?t=${timestamp}`;
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
                        <div>${diameterInfo}</div>
                    </div>
                `;
                
                container.appendChild(title);
                container.appendChild(img);
                container.appendChild(info);
                allContoursContainer.appendChild(container);
            });
            
            // 清空之前的内容并添加新内容
            candidateFrames.innerHTML = '';
            candidateFrames.appendChild(allContoursContainer);
        } else {
            candidateFrames.innerHTML = '<div>未检测到血管边界</div>';
        }
    }

    //-------------------------------------z值计算-------------------------------------
    // 计算Z值功能
    calculateZvalue() {
        // 显示模态框
        const modal = document.getElementById('z-value-modal');
        modal.style.display = 'block';
        
        // 清空之前的结果
        document.getElementById('z-value-result').style.display = 'none';
        document.getElementById('height').value = '';
        document.getElementById('weight').value = '';
    }
}

// 初始化查看器
const viewer = new MedicalImageViewer();

// 添加图像点击事件监听器
const imageDisplay = document.getElementById('image-display');
const measurementOverlay = document.getElementById('measurement-overlay');

// ROI 测量的鼠标事件处理
if (imageDisplay) {
    imageDisplay.addEventListener('mousedown', (event) => {
        if (viewer.measurementMode === 'roi') {
            event.preventDefault();
            viewer.handleMouseDown(event);
            // 当进入 ROI 模式时，启用覆盖层的鼠标事件
            if (measurementOverlay) {
                measurementOverlay.classList.add('roi-mode');
            }
        }
    });
}

if (measurementOverlay) {
    measurementOverlay.addEventListener('mousemove', (event) => {
        if (viewer.measurementMode === 'roi') {
            event.preventDefault();
            viewer.handleMouseMove(event);
        }
    });

    measurementOverlay.addEventListener('mouseup', (event) => {
        if (viewer.measurementMode === 'roi') {
            event.preventDefault();
            viewer.handleMouseUp(event);
            // 当退出 ROI 模式时，禁用覆盖层的鼠标事件
            measurementOverlay.classList.remove('roi-mode');
        }
    });
}

// 防止鼠标在图像外释放时丢失事件
document.addEventListener('mouseup', (event) => {
    if (viewer.measurementMode === 'roi' && viewer.isDrawing) {
        event.preventDefault();
        viewer.handleMouseUp(event);
        // 当退出 ROI 模式时，禁用覆盖层的鼠标事件
        if (measurementOverlay) {
            measurementOverlay.classList.remove('roi-mode');
        }
    }
});

// 手动测量的点击事件
if (imageDisplay) {
    imageDisplay.addEventListener('click', (event) => {
        // 只有在手动测量模式下才处理点击事件
        if (viewer.measurementMode === true) {
            viewer.handleImageClick(event);
        }
    });
}

// 等待DOM加载完成后再绑定事件
document.addEventListener('DOMContentLoaded', function() {
    // 文件列表点击文件名加载对应文件
    document.querySelectorAll('.file-item').forEach(item => {
        item.addEventListener('click', function() {
            const fileName = this.getAttribute('data-file');
            viewer.loadFile(fileName);
        });
    });

    // 绑定按钮事件
    const importBtn = document.getElementById('import-btn');
    const infoBtn = document.getElementById('info-btn');
    const singleMeasureBtn = document.getElementById('single-measure-btn');
    const batchMeasureBtn = document.getElementById('batch-measure-btn');
    const manualMeasureBtn = document.getElementById('manual-measure-btn');
    const roiMeasureBtn = document.getElementById('roi-measure-btn');
    const clearBtn = document.getElementById('clear-btn');
    const zValueBtn = document.getElementById('Z-value');
    const exportBtn = document.getElementById('export-btn');
    const helpBtn = document.getElementById('help-btn');

    if (importBtn) importBtn.onclick = () => viewer.importFile();
    if (infoBtn) infoBtn.onclick = () => viewer.showInfo();
    if (singleMeasureBtn) singleMeasureBtn.onclick = () => viewer.startMeasurement();
    if (batchMeasureBtn) batchMeasureBtn.onclick = () => viewer.autoBatchMeasure();
    if (manualMeasureBtn) manualMeasureBtn.onclick = () => viewer.startManualMeasurement();
    if (roiMeasureBtn) roiMeasureBtn.onclick = () => viewer.drawRoiMeasurement();
    if (clearBtn) clearBtn.onclick = () => viewer.clearMeasurements();
    if (zValueBtn) zValueBtn.onclick = () => viewer.calculateZvalue();
    if (exportBtn) exportBtn.onclick = () => viewer.exportReport();
    if (helpBtn) helpBtn.onclick = () => viewer.showHelp();

    // 获取模态框元素
    const modal = document.getElementById('z-value-modal');
    const span = document.getElementsByClassName('close')[0];

    // 关闭模态框
    if (span) {
        span.onclick = function() {
            if (modal) {
                modal.style.display = 'none';
            }
        }
    }

    // 点击模态框外部关闭
    window.onclick = function(event) {
        if (modal && event.target == modal) {
            modal.style.display = 'none';
        }
    }

    // 计算各血管类型的Z值
    const calculateLcaBtn = document.getElementById('calculate-lca');
    const calculateLadBtn = document.getElementById('calculate-lad');
    const calculateRcaBtn = document.getElementById('calculate-rca');

    if (calculateLcaBtn) {
        calculateLcaBtn.onclick = async function() {
            await calculateSpecificZValue('LCA');
        };
    }

    if (calculateLadBtn) {
        calculateLadBtn.onclick = async function() {
            await calculateSpecificZValue('LAD');
        };
    }

    if (calculateRcaBtn) {
        calculateRcaBtn.onclick = async function() {
            await calculateSpecificZValue('RCA');
        };
    }
});

// 计算特定血管的Z值
async function calculateSpecificZValue(vesselType) {
    const heightInput = document.getElementById('height');
    const weightInput = document.getElementById('weight');
    const measuredValueInput = document.getElementById('measured-value');
    
    if (!heightInput || !weightInput) {
        console.error('找不到身高或体重输入框');
        return;
    }
    
    const height = parseFloat(heightInput.value);
    const weight = parseFloat(weightInput.value);
    const measuredValue = measuredValueInput ? parseFloat(measuredValueInput.value) : null;
    
    if (!height || !weight) {
        alert('请输入有效的身高和体重');
        return;
    }
    
    if (height <= 0 || weight <= 0) {
        alert('身高和体重必须大于0');
        return;
    }
    
    // 如果没有输入实测值，默认使用3.0mm
    const actualMeasuredValue = (measuredValue && measuredValue > 0) ? measuredValue : 3.0;
    
    try {
        const response = await fetch('/calculate-z-value', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                height: height,
                weight: weight,
                measured_value: actualMeasuredValue,
                vessel_type: vesselType
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            const resultDiv = document.getElementById('z-value-result');
            if (resultDiv) {
                resultDiv.textContent = `${vesselType} Z值: ${data.z_value.toFixed(2)}`;
                resultDiv.style.display = 'block';
            }
        } else {
            alert('计算失败: ' + data.error);
        }
    } catch (error) {
        console.error('计算Z值时出错:', error);
        alert('计算过程中发生错误，请查看控制台了解详情');
    }
}