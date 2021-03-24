# ===============================
# PART 2: Decorder
# ===============================

import numpy as np

import tensorflow as tf
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile

import tfcoreml
import coremltools

# The number of predicted classes, excluding background.
num_classes = 90

# The number of predicted bounding boxes.
num_anchors = 1917

# Size of the expected input image.
input_width = 300
input_height = 300


tf_model_path = 'ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb'

## 加载冷冻的tf图
def load_frozenGraph(frozen_graph_filename):
  with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
    return graph_def

load_frozenGraph(tf_model_path)
original_graph = tf.get_default_graph()
## 打印节点
tensorname_list = []
for tensor in original_graph.as_graph_def().node:
  tensor_name = tensor.name
  tensorname_list.append(tensor_name)
  #if "Concatenate" in tensor_name:
  #print(tensor_name, '\n')
  #if "concat" in tensor_name:
  print(tensor_name, '\n')
  if tensor_name == "Postprocessor/ExpandDims/_236__cf__239":
    #print(tensor_name, '\n')
    print(tensor)
#print(tensorname_list)



# ================================
# PART 2: Decoding the coordinates
# ================================

def get_anchors(graph, tensor_name):
    """
    Computes the list of anchor boxes by sending a fake image through the graph.
    Outputs an array of size (4, num_anchors) where each element is an anchor box
    given as [ycenter, xcenter, height, width] in normalized coordinates.
    """
    image_tensor = graph.get_tensor_by_name("image_tensor:0")
    box_corners_tensor = graph.get_tensor_by_name(tensor_name)
    #输入一个假的图片（0像素矩阵）,得到产生的1917个锚框（任何图片都一样）:anchor box coordinates are normalized,
    box_corners = sess.run(box_corners_tensor, feed_dict={image_tensor: np.zeros((1, input_height, input_width, 3))})
    print('box_corners.shape: ', box_corners.shape)
    # The TensorFlow graph gives each anchor box as [ymin, xmin, ymax, xmax]. 
    # Convert these min/max values to a center coordinate, width and height.
    ymin, xmin, ymax, xmax = np.transpose(box_corners)
    print('ymin: ', ymin.shape)
    width = xmax - xmin
    height = ymax - ymin
    ycenter = ymin + height / 2.
    xcenter = xmin + width / 2.
    # 按行将向量组合为矩阵： [4, 1917, 1]
    return np.stack([ycenter, xcenter, height, width])



# Read the anchors into a (4, 1917) tensor.(输出的1917个锚框：每个锚框由[ycenter, xcenter, height, width]标识)
anchors_tensor = "Postprocessor/ExpandDims/_236__cf__239:0"

with original_graph.as_default():
    with tf.Session(graph=original_graph) as sess:
      anchors = get_anchors(original_graph, anchors_tensor)
      print('anchors.shape', anchors.shape)#[4, 1917, 1]
      assert(anchors.shape[1] == num_anchors)


from coremltools.models import datatypes
from coremltools.models import neural_network

# MLMultiArray inputs of neural networks must have 1 or 3 dimensions. 
# We only have 2, so add an unused dimension of size one at the back.
# scores: [91, 1917, 1]-- width==91, hight==1917, channel == 1
# boxes: [4, 1917, 1]--

input_features = [ ("boxes", datatypes.Array(4, num_anchors, 1)), 
				("scores", datatypes.Array(num_classes + 1, num_anchors, 1))
                   ]

# The outputs of the decoder model should match the inputs of the next
# model in the pipeline, NonMaximumSuppression. This expects the number
# of bounding boxes in the first dimension.
output_features = [ ("raw_confidence", datatypes.Array(num_anchors, num_classes)),
                    ("raw_coordinates", datatypes.Array(num_anchors, 4)) ]


#====================== 处理类别信息：置信度scores ===============================#
builder = neural_network.NeuralNetworkBuilder(input_features, output_features)

# 例--维度变换[seq, C, H, W]-->dim:[0, 1, 2, 3]
# 			dim:[0, 3, 1, 2]-->[seq, W, C, H]
# [seq,91,1917,1]：(num_classes+1, num_anchors, 1) --> [1, 1917, 91](1, num_anchors, num_classes+1)
builder.add_permute(name="permute_scores",
                    dim=(0, 3, 2, 1),
                    input_name="scores",
                    output_name="permute_scores_output")

#input: [1, 1917, 91]: 取出第一行到第九十行（第0行剔除1917）
#output: [1, 1917, 90]
# Strip off the "unknown" class (at index 0).: 支持[‘channel’, ‘height’, ‘width’] 
# width = 91
builder.add_slice(name="slice_scores",
                  input_name="permute_scores_output",
                  output_name="raw_confidence",
                  axis="width",
                  start_index=1,
                  end_index=num_classes + 1)


#====================== 处理bbox信息：位置及框选 ===============================#
# input: boxes: [4, 1917, 1]   [‘channel’, ‘height’, ‘width’]
# 0utput: [2, 1917, 1]
# channel = 4
# Grab the y, x coordinates [ channels 0-1:(ycenter, xcenter) ].
builder.add_slice(name="slice_yx",
                  input_name="boxes",
                  output_name="slice_yx_output",
                  axis="channel",
                  start_index=0,
                  end_index=2)

# boxes_yx / 10
builder.add_elementwise(name="scale_yx",
                        input_names="slice_yx_output",
                        output_name="scale_yx_output",
                        mode="MULTIPLY",
                        alpha=0.1)

# Split the anchors into two (2, 1917, 1) arrays.
anchors_yx = np.expand_dims(anchors[:2, :], axis=-1)
anchors_hw = np.expand_dims(anchors[2:, :], axis=-1)


builder.add_load_constant(name="anchors_yx",
                          output_name="anchors_yx",
                          constant_value=anchors_yx,
                          shape=[2, num_anchors, 1])

builder.add_load_constant(name="anchors_hw",
                          output_name="anchors_hw",
                          constant_value=anchors_hw,
                          shape=[2, num_anchors, 1])

# (boxes_yx / 10) * anchors_hw
builder.add_elementwise(name="yw_times_hw",
                        input_names=["scale_yx_output", "anchors_hw"],
                        output_name="yw_times_hw_output",
                        mode="MULTIPLY")

# (boxes_yx / 10) * anchors_hw + anchors_yx
builder.add_elementwise(name="decoded_yx",
                        input_names=["yw_times_hw_output", "anchors_yx"],
                        output_name="decoded_yx_output",
                        mode="ADD")


# input: boxes: [4, 1917, 1]   [‘channel’, ‘height’, ‘width’]
# 0utput: [2, 1917, 1]
# channel = 4
# Grab the height and width (channels 2-3:[height, width]).
builder.add_slice(name="slice_hw",
                  input_name="boxes",
                  output_name="slice_hw_output",
                  axis="channel",
                  start_index=2,
                  end_index=4)

# (boxes_hw / 5)
builder.add_elementwise(name="scale_hw",
                        input_names="slice_hw_output",
                        output_name="scale_hw_output",
                        mode="MULTIPLY",
                        alpha=0.2)

# exp(boxes_hw / 5)
builder.add_unary(name="exp_hw",
                  input_name="scale_hw_output",
                  output_name="exp_hw_output",
                  mode="exp")

# exp(boxes_hw / 5) * anchors_hw
builder.add_elementwise(name="decoded_hw",
                        input_names=["exp_hw_output", "anchors_hw"],
                        output_name="decoded_hw_output",
                        mode="MULTIPLY")


# The coordinates are now (y, x) and (height, width) but NonMaximumSuppression
# wants them as (x, y, width, height). So create four slices and then concat
# them into the right order.
# 取出y
builder.add_slice(name="slice_y",
                  input_name="decoded_yx_output",
                  output_name="slice_y_output",
                  axis="channel",
                  start_index=0,
                  end_index=1)

# 取出x
builder.add_slice(name="slice_x",
                  input_name="decoded_yx_output",
                  output_name="slice_x_output",
                  axis="channel",
                  start_index=1,
                  end_index=2)

# 取出h
builder.add_slice(name="slice_h",
                  input_name="decoded_hw_output",
                  output_name="slice_h_output",
                  axis="channel",
                  start_index=0,
                  end_index=1)

# 取出w
builder.add_slice(name="slice_w",
                  input_name="decoded_hw_output",
                  output_name="slice_w_output",
                  axis="channel",
                  start_index=1,
                  end_index=2)

# 合并[x,y,w,h]
builder.add_elementwise(name="concat",
                        input_names=["slice_x_output", "slice_y_output", 
                                     "slice_w_output", "slice_h_output"],
                        output_name="concat_output",
                        mode="CONCAT")

# bbox维度变换
# (4, num_anchors, 1) --> (1, num_anchors, 4)
builder.add_permute(name="permute_output",
                    dim=(0, 3, 2, 1),
                    input_name="concat_output",
                    output_name="raw_coordinates")

decoder_model = coremltools.models.MLModel(builder.spec)
decoder_model.save("Decoder.mlmodel")