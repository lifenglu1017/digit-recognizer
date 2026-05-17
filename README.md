# CNN 手写数字识别系统

基于PyTorch和Gradio的手写数字识别Web应用。

## 📁 项目结构

```
digit-recognizer/
├── app.py              # Gradio Web应用主程序
├── model.pth           # 训练好的CNN模型权重
├── requirements.txt    # Python依赖包
├── train.py            # 模型训练脚本
├── README.md           # 项目说明文档
└── .gitignore          # Git忽略文件配置
```

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/yourusername/digit-recognizer.git
cd digit-recognizer
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 运行应用
```bash
python app.py
```

### 4. 访问界面
打开浏览器访问：**http://localhost:7860**

## 📝 使用说明

- 在画板上用鼠标绘制0-9的数字
- 系统会自动识别并显示Top3概率结果

## 🛠️ 技术栈

- PyTorch 2.11.0
- Torchvision 0.26.0
- Gradio 6.14.0
- Python 3.13

## 📄 许可证

MIT License
