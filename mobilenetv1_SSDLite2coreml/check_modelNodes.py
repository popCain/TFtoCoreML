# ==============================================================
# PART 1: 
#         ・加载冷冻模型（原全网络：'preprocess' 
#                               'mobilenetv2+ssdlite' + 'anchor generate' + 'NMS' + 'predictor'）
#         ・简化原冷冻模型（提取出'mobilenetv2+ssdlite'网络部分）
#          ・简化后的网络转化为‘mobilenetv2+ssdlite.mlmodel’
# ==============================================================
import tensorflow as tf


#输入-- mobilenetv3网络的输入
input_node = "Preprocessor/sub"

#输出--output[0]:raw_outputs/box_encodings(); output[1]:raw_outputs/box_encodings
class_output_node = "detection_scores"
bbox_output_node = "detection_boxes"
num_detections = "num_detections"


# ==============================================================
# fun load_frozenGraph: 加载冷冻模型
#     （最初始的完整网络： 
#             'mobilenetv3+ssdlite'）
# ==============================================================
tf_model_path = 'ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb'
def load_frozenGraph(frozen_graph_filename):
  with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
    return graph_def

#获得的原冻图
original_gdef = load_frozenGraph(tf_model_path)

## 打印节点
for tensor in tf.get_default_graph().as_graph_def().node:
  tensor_name = tensor.name
  print(tensor_name, '\n')
  
  if tensor_name == "anchors":
    print(tensor_name, '\n')
    print(tensor.attr)