# ==============================================================
# 转化为MLModel
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

# mobilenetv1 + ssd网络结构模型
coreml_model_path = 'mobilenetV1_SSD.mlmodel'

#输入-- mobilenetv1网络的输入
input_node = "Preprocessor/sub"

#输出-- concat : bbox(coordinate) / Postprocessor/convert_scores: class(score)
bbox_output_node = "concat"
class_output_node = "Postprocessor/convert_scores"

#输入输出转化为tensor形式的变量
input_tensor = input_node + ":0"
bbox_output_tensor = bbox_output_node + ":0"
class_output_tensor = class_output_node + ":0"

frozen_model_file = 'ssd_mobilenet.pb'
#[-1,1]
ssd_model = tfcoreml.convert(
	tf_model_path = frozen_model_file,
	mlmodel_path = coreml_model_path,
	input_name_shape_dict = {
	input_tensor:[1, input_height, input_width, 3]},
	image_input_names = input_tensor,
	output_feature_names = [bbox_output_tensor, class_output_tensor],
	is_bgr = False,
	red_bias = -1.0,
	green_bias = -1.0,
	blue_bias = -1.0,
	image_scale = 2./255)


spec = ssd_model.get_spec()
## Preprocessor__sub__0, concat__0, Postprocessor__convert_scores__0
## 转换时是自动将“/”和“：”变换成了“__”,所以我们根据其形式转换自己的protobuf（各层的名称操作）
#print(spec.description)
# Rename the inputs and outputs to something more readable.
spec.description.input[0].name = "image"
spec.description.input[0].shortDescription = "Input image"
spec.description.output[0].name = "scores"
spec.description.output[0].shortDescription = "Predicted class scores for each bounding box"
spec.description.output[1].name = "boxes"
spec.description.output[1].shortDescription = "Predicted coordinates for each bounding box"


input_mlmodel = input_tensor.replace(":", "__").replace("/", "__")
class_output_mlmodel = class_output_tensor.replace(":", "__").replace("/", "__")
bbox_output_mlmodel = bbox_output_tensor.replace(":", "__").replace("/", "__")

for i in range(len(spec.neuralNetwork.layers)):
    if spec.neuralNetwork.layers[i].input[0] == input_mlmodel:
        spec.neuralNetwork.layers[i].input[0] = "image"
    if spec.neuralNetwork.layers[i].output[0] == class_output_mlmodel:
        spec.neuralNetwork.layers[i].output[0] = "scores"
    if spec.neuralNetwork.layers[i].output[0] == bbox_output_mlmodel:
        spec.neuralNetwork.layers[i].output[0] = "boxes"

spec.neuralNetwork.preprocessing[0].featureName = "image"

# For some reason the output shape of the "scores" output is not filled in.
spec.description.output[0].type.multiArrayType.shape.append(num_classes + 1)
spec.description.output[0].type.multiArrayType.shape.append(num_anchors)


# And the "boxes" output shape is (4, 1917, 1) so get rid of that last one.
del spec.description.output[1].type.multiArrayType.shape[-1]

# Convert weights to 16-bit floats to make the model smaller.
spec = coremltools.utils.convert_neural_network_spec_weights_to_fp16(spec)

# Create a new MLModel from the modified spec and save it.
ssd_model = coremltools.models.MLModel(spec)
ssd_model.save(coreml_model_path)
