import json
import csv
import base64
import os
import re

class TGDatasetBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.json_file_path = cfg.get('json_file_path', 'result.json')
        self.output_csv_path = cfg.get('output_csv_path', 'dataset.csv')
        self.image_search_path = cfg.get('image_search_path', '.')
        self.data = None
    
    def load_data(self):
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            return True
        except Exception as e:
            print(f"Ошибка при загрузке JSON файла: {e}")
            return False
    
    def find_image_file(self, filename):
        for root, dirs, files in os.walk(self.image_search_path):
            if filename in files:
                return os.path.join(root, filename)
        return None
    
    def image_to_base64(self, image_path):
        try:
            with open(image_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
        except Exception as e:
            print(f"Ошибка при чтении файла {image_path}: {e}")
            return None
    
    def clean_text(self, text):
        text = re.sub(r"[\n\r\t]+", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()
    
    def extract_description(self, txt_ctx):
        desc = ""
        
        if isinstance(txt_ctx, str):
            desc = txt_ctx
        elif isinstance(txt_ctx, list):
            for item in txt_ctx:
                if isinstance(item, str):
                    desc += item
                elif isinstance(item, dict) and item.get('type') == 'plain' and 'text' in item:
                    desc += item['text']
        
        return self.clean_text(desc)
    
    def process_messages(self):
        memes = []
        
        if not self.data:
            return memes
        
        for msg in self.data['messages']:
            if 'text' in msg and 'photo' in msg and msg['text']:
                desc = self.extract_description(msg['text'])
                
                if not desc:
                    continue
                
                image_name = os.path.basename(msg['photo'])
                image_path = self.find_image_file(image_name)
                
                if image_path:
                    base64_image = self.image_to_base64(image_path)
                    
                    if base64_image:
                        record = {
                            'image': base64_image,
                            'description': desc,
                            'width': msg['width'],
                            'height': msg['height'],
                            'image_link': 'none'
                        }
                        memes.append(record)
        
        return memes
    
    def build_dataset(self):
        if not self.load_data():
            return 0
        
        memes = self.process_messages()
        
        if not memes:
            return 0
        
        try:
            with open(self.output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['image', 'description', 'width', 'height', 'image_link']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                
                for meme in memes:
                    writer.writerow(meme)
            
            return True
            
        except Exception as e:
            print(f"Ошибка при создании CSV файла: {e}")
            return 0

if __name__ == "__main__":
    cfg = {
        'json_file_path': 'result.json',
        'output_csv_path': 'tg_dataset.csv',
        'image_search_path': '.'
    }
    
    builder = TGDatasetBuilder(cfg)
    success = builder.build_dataset()