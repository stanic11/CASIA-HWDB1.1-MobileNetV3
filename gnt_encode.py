import struct
import os
import re
import numpy as np
from PIL import Image

class gntDecoder:
    def __init__(self, params):
        self.params = params
        self.train_label_map = {}  # 保存训练集的字符->标签编号
        self.process_datasets()

    def process_datasets(self):
        # 处理训练集
        self.train_label_map = self.process_dataset(
            gnt_dir=self.params["train_gnt_dir"],
            data_dir=self.params["train_data_dir"],
            is_train=True)
        # 处理测试集
        self.process_dataset(
            gnt_dir=self.params["test_gnt_dir"],
            data_dir=self.params["test_data_dir"],
            is_train=False,
            label_map=self.train_label_map)
        
    def clean_filename(self, filename):
        return re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    def is_chinese_char(self, char):
        cp = ord(char)
        return (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF)

    def process_dataset(self, gnt_dir, data_dir, is_train, label_map=None):
        if is_train:
            label_map = {}

        total_images = 0
        files = os.listdir(gnt_dir)
        for file_idx, filename in enumerate(files):
            file_path = os.path.join(gnt_dir, filename)
            print(f"Processing {file_idx+1}/{len(files)}: {filename}")
            
            with open(file_path, 'rb') as f:
                while True:
                    header = f.read(4)
                    if not header:
                        break
                    
                    tag_code = f.read(2)
                    width = struct.unpack('<H', f.read(2))[0]
                    height = struct.unpack('<H', f.read(2))[0]
                    image_data = f.read(width * height)
                    
                    try:
                        char = tag_code.decode('gb18030').strip('\x00')
                    except UnicodeDecodeError:
                        continue
                    
                    if len(char) != 1 or not self.is_chinese_char(char):
                        continue
                    
                    clean_char = self.clean_filename(char)
                    if is_train and clean_char not in label_map:
                        label_map[clean_char] = len(label_map) + 1
                    
                    num_label = label_map[clean_char]
                    target_dir = os.path.join(data_dir, str(num_label))
                    os.makedirs(target_dir, exist_ok=True)
                    
                    img_path = os.path.join(target_dir, f"{total_images}.png")
                    Image.frombytes('L', (width, height), image_data).save(img_path)
                    total_images += 1

        if is_train:
            self._save_mapping_matrix(data_dir, label_map)
            print(f"[训练集] 处理完成：{total_images} 张图片，{len(label_map)} 个类别")
            return label_map
        else:
            print(f"[测试集] 处理完成：{total_images} 张图片")
            return None

    def _save_mapping_matrix(self, data_dir, label_map):
        dtype = np.dtype([('num', 'i2'), ('char', 'U1')])
        sorted_items = sorted(label_map.items(), key=lambda x: x[1])
        mapping_data = np.array(
            [(num, char) for char, num in sorted_items],
            dtype=dtype
        )
        np.save(os.path.join(data_dir, 'mapping_matrix.npy'), mapping_data, allow_pickle=False)
        print(f"保存映射矩阵至 {os.path.join(data_dir, 'mapping_matrix.npy')}")
