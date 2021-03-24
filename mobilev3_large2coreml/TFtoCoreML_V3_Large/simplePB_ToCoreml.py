# =========================================================================
# PART2:
#		转化为MLModel:
# 		20*20*3(1200) 10*10*6(600) 5*5*6(150) 3*3*6(54) 2*2*6(24) 1*1*6(6)
# 		共计2034个预测框（prior anchors）
# =========================================================================
import tensorflow as tf
import tfcoreml
import coremltools
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile
from coremltools.models import datatypes


# The number of predicted classes, excluding background.
num_classes = 90

# The number of predicted bounding boxes.
num_anchors = 2034

# Size of the expected input image.
input_width = 320
input_height = 320

# mobilenetv3 + ssd网络(coreml文件存储路径)
coreml_model_path = 'ssd_mobilenet.mlmodel'

#输入-- mobilenetv3网络的输入
input_node = "Preprocessor/sub"

#输出-- Postprocessor/raw_box_scores: class(score) // Postprocessor/raw_box_encodings: BBOX(coordinate)
class_output_node = "Postprocessor/raw_box_scores"
bbox_output_node = "Postprocessor/raw_box_encodings"


#输入输出转化为tensor形式的变量
input_tensor = input_node 
class_output_tensor = class_output_node 
bbox_output_tensor = bbox_output_node 
 
# OP:Squeeze(去掉不需要维度)--squeeze_dims = 2 : (1, 2034, 1, 4) ->> (1, 2034, 4)
frozen_model_file = 'ssd_mobilenet.pb'
# 已经能够产生mlmodel了
ssd_model = tfcoreml.convert(
	tf_model_path = frozen_model_file,
	mlmodel_path = coreml_model_path,
	input_name_shape_dict = {input_tensor:[1, input_height, input_width, 3]},
	image_input_names = input_tensor,
	output_feature_names = [class_output_tensor, bbox_output_tensor],
	is_bgr = False,
	red_bias = -1.0,
	green_bias = -1.0,
	blue_bias = -1.0,
	image_scale = 2./255,
	# MobilenetV3/Conv/BatchNorm/FusedBatchNormV3を使えるようの為
	minimum_ios_deployment_target='13'
	)

# 获取转化后coreml文件的描述：protobuf(Google)
spec = ssd_model.get_spec()
print(spec.description)
print("*"*20 + "multiArrayType维度指定后" + "*"*20)


# =========================================================================
#	原输出形式: multiArrayType {dataType: FLOAT32}
#	根据实际输出的维度，明确指定:
#			output[0]: Postprocessor/raw_box_scores --> [1, 2034, 91]
#			output[1]: Postprocessor/raw_box_encodings --> [1, 2034, 4]
# =========================================================================
spec.description.output[0].type.multiArrayType.shape.append(1)
spec.description.output[0].type.multiArrayType.shape.append(num_anchors)
spec.description.output[0].type.multiArrayType.shape.append(num_classes + 1)

spec.description.output[1].type.multiArrayType.shape.append(1)
spec.description.output[1].type.multiArrayType.shape.append(num_anchors)
spec.description.output[1].type.multiArrayType.shape.append(4)


def update_multiarray_to_Double(feature):  
    if feature.type.HasField('multiArrayType'):  
        import coremltools.proto.FeatureTypes_pb2 as _ft  
        feature.type.multiArrayType.dataType = _ft.ArrayFeatureType.DOUBLE

for input_feature in spec.description.input:  
    update_multiarray_to_Double(input_feature)  
  
for output_feature in spec.description.output:  
    update_multiarray_to_Double(output_feature)

# Convert weights to 16-bit floats to make the model smaller.
spec = coremltools.utils.convert_neural_network_spec_weights_to_fp16(spec)
print(spec.description)

ssd_model = coremltools.models.MLModel(spec)
ssd_model.save(coreml_model_path)
