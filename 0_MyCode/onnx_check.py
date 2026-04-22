import numpy as np
import onnxruntime
from PIL import Image

##### 固定配置 #####
img_path = (
    "/home/wapiti/Projects/Anomaly_D/Datasets/MVTecAD/bottle/test/broken_large/000.png"
)
onnx_file_name = "/home/wapiti/Projects/Anomaly_D/stfpm.onnx"
##### 固定配置 #####

img = Image.open(img_path).convert("RGB").resize((256, 256))
array = np.array(img).astype(np.float32)
normalize = array / 255.0
img_chw = normalize.transpose(2, 0, 1)
img = np.expand_dims(img_chw, 0)

ort_session = onnxruntime.InferenceSession(onnx_file_name)
print(ort_session.get_inputs()[0].name)
print(ort_session.get_inputs()[0].shape)

# print(type(ort_session.get_outputs()))

# for idx,output_meta in enumerate(ort_session.get_outputs()):
#     # print(f"{idx}:{output_meta}")
#     print(f"{idx}:{ort_session.get_outputs()[idx].name},{ort_session.get_outputs()[idx].shape}")

ort_inputs = {ort_session.get_inputs()[0].name: img}
ort_outs = ort_session.run(None, ort_inputs)

score = None
anomaly_map_tensor = None

for output_meta, output_value in zip(ort_session.get_outputs(), ort_outs):
    # print(f"{output_meta.name},{output_value.shape}")
    if output_meta.name == "output":
        score = float(output_value)
        print(f"score:{score}")
    elif output_meta.name == "anomaly_map":
        anomaly_map_tensor = np.squeeze(output_value)
        print(f"anomaly_map_tensor:{anomaly_map_tensor.shape}")
        print(f"min:{anomaly_map_tensor.min()}")
        print(f"max:{anomaly_map_tensor.max()}")


""" 
Evaluation mode:
InferenceBatch:
Batch containing anomaly maps and prediction scores.
ort_session.get_outputs() 是说明书
告诉你每个输出叫什么、形状应该是什么

ort_outs 是实物
是模型真正算出来的数据
 """
