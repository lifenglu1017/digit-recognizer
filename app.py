import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
import numpy as np

print(f"Gradio版本: {gr.__version__}", flush=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc(x)
        return x

print("加载模型...", flush=True)
model = CNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu'), weights_only=False))
model.eval()
print("模型加载成功!", flush=True)

def center_digit(img):
    img_np = np.array(img)
    
    rows = np.any(img_np < 255, axis=1)
    cols = np.any(img_np < 255, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return img
    
    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]
    
    cropped = img_np[min_row:max_row+1, min_col:max_col+1]
    
    size = max(cropped.shape[0], cropped.shape[1])
    square = np.ones((size, size), dtype=np.uint8) * 255
    
    y_offset = (size - cropped.shape[0]) // 2
    x_offset = (size - cropped.shape[1]) // 2
    
    square[y_offset:y_offset+cropped.shape[0], x_offset:x_offset+cropped.shape[1]] = cropped
    
    return Image.fromarray(square)

transform = transforms.Compose([
    transforms.Resize((20, 20)),
    transforms.Pad(4, fill=255),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def predict_digit(image_data):
    try:
        if image_data is None:
            return {"请绘制数字": 1.0}

        if isinstance(image_data, dict):
            if 'layers' in image_data and len(image_data['layers']) > 0:
                image_data = image_data['layers'][0]
        
        if isinstance(image_data, np.ndarray):
            img = Image.fromarray(image_data)
        elif isinstance(image_data, Image.Image):
            img = image_data
        else:
            raise ValueError(f"不支持的图像类型: {type(image_data)}")

        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        img = img.convert('L')
        img = img.point(lambda x: 255 - x)
        img = center_digit(img)
        
        img_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.softmax(output, dim=1)[0]

        result = {str(i): float(prob[i]) for i in range(10)}
        print(f"识别结果: {result}", flush=True)
        return result
        
    except Exception as e:
        print(f"识别错误: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {"错误": 1.0}

print("创建界面...", flush=True)

inputs = gr.Sketchpad(
    height=280,
    width=280
)

outputs = gr.Label(num_top_classes=3)

print("启动服务...", flush=True)
if __name__ == "__main__":
    gr.Interface(
        fn=predict_digit,
        inputs=inputs,
        outputs=outputs,
        title="🎨 CNN手写数字识别",
        description="在画板上用鼠标绘制0-9的数字，系统将实时识别并显示Top3概率",
        examples=None,
        cache_examples=False
    ).launch(server_port=7860, share=False)