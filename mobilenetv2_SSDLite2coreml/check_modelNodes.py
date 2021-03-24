# ==============================================================
# PART 1: 
#         ・加载冷冻模型（原全网络：'preprocess' 
#                               'mobilenetv2+ssdlite' + 'anchor generate' + 'NMS' + 'predictor'）
#         ・简化原冷冻模型（提取出'mobilenetv2+ssdlite'网络部分）
#          ・简化后的网络转化为‘mobilenetv2+ssdlite.mlmodel’
# ==============================================================
import tensorflow as tf

# ==============================================================
# fun load_frozenGraph: 加载冷冻模型
#     （最初始的完整网络： 
#             'mobilenetv2+ssdlite'）
# ==============================================================
tf_model_path = 'ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb'
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