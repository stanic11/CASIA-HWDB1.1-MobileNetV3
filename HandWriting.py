import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import torch
from torchvision import transforms
from MobileNet import MobileNetV3
import os
from main import ResizePad  

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("手写输入识别")

        main_frame = tk.Frame(root)
        main_frame.pack(padx=10, pady=10)

        self.canvas_size = 280  # 画板尺寸
        self.model_input_size = 96  # 模型输入尺寸

        self.canvas = tk.Canvas(main_frame, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.canvas.grid(row=0, column=0, rowspan=3, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.draw)

        output_frame = tk.LabelFrame(main_frame, text="识别结果", padx=10, pady=10)
        output_frame.grid(row=0, column=1, sticky="n", padx=10)
        self.output_label = tk.Label(output_frame, text="类别：\n汉字：", font=("微软雅黑", 16), justify="left")
        self.output_label.pack()

        btn_frame = tk.Frame(main_frame)
        btn_frame.grid(row=1, column=1, sticky="n")
        tk.Button(btn_frame, text="清空", command=self.clear_canvas, width=10).pack(pady=5)
        tk.Button(btn_frame, text="识别", command=self.predict, width=10).pack(pady=5)

        self.points = []

        # 模型和设备初始化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MobileNetV3(num_classes=100, in_channels=1)
        self.model.load_state_dict(torch.load('weights/MobileNetV3.pth', map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            ResizePad(size=self.model_input_size, fill=255),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.mapping = None
        self.load_mapping_matrix()

    def load_mapping_matrix(self):
        mapping_path = os.path.join(os.getcwd(), "mapping_matrix.npy")
        if os.path.exists(mapping_path):
            self.mapping = np.load(mapping_path, allow_pickle=True)
        else:
            self.mapping = None

    def draw(self, event):
        x, y = event.x, event.y
        self.points.append((x, y))
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.points = []
        self.output_label.config(text="类别：\n汉字：")

    def preprocess_image(self):
        img = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        draw = ImageDraw.Draw(img)
        for (x, y) in self.points:
            draw.ellipse((x-5, y-5, x+5, y+5), fill='black')
        img = self.transform(img)  # transforms 自动处理为128x128
        img = img.unsqueeze(0)  # [1, 1, 96, 96]
        return img.to(self.device)

    def predict(self):
        if not self.points:
            self.output_label.config(text="类别：\n汉字：")
            return
        input_tensor = self.preprocess_image()
        with torch.no_grad():
            output = self.model(input_tensor)
            pred = output.argmax(dim=1).item()
        hanzi = ""
        if self.mapping is not None and 0 <= pred < len(self.mapping):
            hanzi = str(self.mapping[pred])
        else:
            hanzi = "(未找到映射)"
        self.output_label.config(text=f"类别：{pred}\n汉字：{hanzi}")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()