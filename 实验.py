from modelscope import ViTImageProcessor, ViTModel
from PIL import Image

image = Image.open("C:/Users/ZY/Desktop/123.jpg")

# 使用本地路径替代在线模型名称
local_model_path = r"C:\Users\ZY\.cache\modelscope\hub\models\google\vit-base-patch16-224-in21k"
processor = ViTImageProcessor.from_pretrained(local_model_path)
model = ViTModel.from_pretrained(local_model_path)

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state  #pooler_output 全局语义(无位置信息)

print(last_hidden_states.shape)
print(last_hidden_states)
