# ==============================================================
# PART 1: 
#         ・加载冷冻模型（原全网络：'preprocess' 
#                               'mobilenetv2+ssdlite' + 'anchor generate' + 'NMS' + 'predictor'）
#         ・简化原冷冻模型（提取出'mobilenetv2+ssdlite'网络部分）
#          ・简化后的网络转化为‘mobilenetv2+ssdlite.mlmodel’
# ==============================================================
import tensorflow as tf
import tfcoreml
import coremltools
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile

# The number of predicted classes, excluding background.
num_classes = 90

# The number of predicted bounding boxes.
num_anchors = 1917

# Size of the expected input image.
input_width = 300
input_height = 300

# mobilenetv2 + ssd网络结构模型
coreml_model_path = 'ssd_mobilenet.mlmodel'

#输入-- mobilenetv2网络的输入
input_node = "Preprocessor/sub"

#输出-- concat : bbox(coordinate) / Postprocessor/convert_scores: class(score)
bbox_output_node = "concat"
class_output_node = "Postprocessor/convert_scores"
#num_detections = "num_detections"
#输入输出转化为tensor形式的变量
input_tensor = input_node + ":0"
bbox_output_tensor = bbox_output_node + ":0"
class_output_tensor = class_output_node + ":0"
#um_detections_tensor = num_detections + ":0"
# ==============================================================
# fun load_frozenGraph: 加载冷冻模型
#     （最初始的完整网络： 
#             'preprocess' 
#             'mobilenetv2+ssdlite'
#             'anchor generate' 
#             'NMS' 
#             'predictor'）
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
tensorname_list = []
for tensor in tf.get_default_graph().as_graph_def().node:
  tensor_name = tensor.name
  if "FusedBatchNormV3" in tensor_name:
    print(tensor_name, '\n')
  tensorname_list.append(tensor_name)
  if tensor_name == "image_tensor":
    print(tensor_name, '\n')
    print(tensor)

    

# ==============================================================
# fun sampley_graph: 简化原模型
#     仅保留（MobileNetV2+SSDLite），保存为新的冷冻网络
# ==============================================================
frozen_model_file = 'ssd_mobilenet.pb'
def sampley_graph(graph):
	gdef = strip_unused_lib.strip_unused(
        input_graph_def = graph,
        input_node_names = [input_node],
        output_node_names = [bbox_output_node, class_output_node],
        placeholder_type_enum = dtypes.float32.as_datatype_enum)
	# Save the feature extractor to an output file
	
	with gfile.GFile(frozen_model_file, "wb") as f:
		f.write(gdef.SerializeToString())
##简化图
sampley_graph(original_gdef)	


