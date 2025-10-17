// viewer.js
class MedicalImageViewer {
    constructor() {
        this.currentFile = null;
        this.currentFrame = 0;
        this.isPlaying = false;
        this.measurementMode = false;
        this.pixelSpacing = null;
        this.unit = null;
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
            overlay.innerHTML = '<div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: white; font-size: 16px;">测量工具</div>';
            this.measurementMode = true;
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
    }

    // 导入文件
    importFile() {
        // 实现文件导入逻辑
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.dcm,.dicom,.jpg,.png';
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
                        fileList.innerHTML = '';

                        // 动态添加新文件项
                        data.frames.forEach(frameName => {
                            const li = document.createElement('li');
                            li.className = 'file-item';
                            li.setAttribute('data-file', frameName.split('.')[0]); // 提取文件名
                            li.textContent = frameName;
                            li.onclick = () => this.loadFrame(frameName);
                            fileList.appendChild(li);
                        });

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
}

// 初始化查看器
const viewer = new MedicalImageViewer();

// 绑定事件
document.querySelectorAll('.file-item').forEach(item => {
    item.addEventListener('click', function() {
        const fileName = this.getAttribute('data-file');
        viewer.loadFile(fileName);
    });
});

// 绑定按钮事件
document.querySelector('.menu-button:nth-child(1)').onclick = () => viewer.importFile();
document.querySelector('.menu-button:nth-child(2)').onclick = () => viewer.showInfo();
document.querySelector('.menu-button:nth-child(3)').onclick = () => viewer.manualMeasure();
document.querySelector('.menu-button:nth-child(4)').onclick = () => viewer.exportReport();
document.querySelector('.menu-button:nth-child(5)').onclick = () => viewer.showHelp();
document.querySelector('.menu-button:nth-child(6)').onclick = () => viewer.sortByScore();

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