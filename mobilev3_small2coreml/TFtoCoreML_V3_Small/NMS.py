import coremltools
import numpy as np

# ===============================
# PART 3: Non-maximum suppression
# ===============================
# The number of predicted classes, excluding background.
num_classes = 90

# The number of predicted bounding boxes.
num_anchors = 2034

# Size of the expected input image.
input_width = 320
input_height = 320

mlmodel_path = 'Decoder.mlmodel'
decoder_model = coremltools.models.MLModel(mlmodel_path)

nms_spec = coremltools.proto.Model_pb2.Model()
nms_spec.specificationVersion = 3

# 初始化非极大值抑制层的输入，输出
for i in range(2):
    decoder_output = decoder_model._spec.description.output[i].SerializeToString()

    nms_spec.description.input.add()
    nms_spec.description.input[i].ParseFromString(decoder_output)

    nms_spec.description.output.add()
    nms_spec.description.output[i].ParseFromString(decoder_output)

#输出改名    
nms_spec.description.output[0].name = "confidence"
nms_spec.description.output[1].name = "coordinates"
#输出shape更改
output_sizes = [num_classes, 4]
for i in range(2):
    ma_type = nms_spec.description.output[i].type.multiArrayType
    #FeatureType > multiArrayType > shapeRange > sizeRanges
    print(ma_type)
    ma_type.shapeRange.sizeRanges.add()
    ma_type.shapeRange.sizeRanges[0].lowerBound = 0
    ma_type.shapeRange.sizeRanges[0].upperBound = -1
    ma_type.shapeRange.sizeRanges.add()
    ma_type.shapeRange.sizeRanges[1].lowerBound = output_sizes[i]
    ma_type.shapeRange.sizeRanges[1].upperBound = output_sizes[i]
    # 去掉其中的shape，添加了shapeRange.sizeRanges((lowerBound,upperBound):(lowerBound,upperBound))
    del ma_type.shape[:]
    print('size:', ma_type.shapeRange.sizeRanges)
print(nms_spec.description.output)
# protobuf (message首字母小写)
nms = nms_spec.nonMaximumSuppression
nms.confidenceInputFeatureName = "raw_confidence"
nms.coordinatesInputFeatureName = "raw_coordinates"
nms.confidenceOutputFeatureName = "confidence"
nms.coordinatesOutputFeatureName = "coordinates"
nms.iouThresholdInputFeatureName = "iouThreshold"
nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
#0.4 0.5
default_iou_threshold = 0.45
default_confidence_threshold = 0.5
nms.iouThreshold = default_iou_threshold
nms.confidenceThreshold = default_confidence_threshold

nms.pickTop.perClass = True

labels = np.loadtxt("coco_labels.txt", dtype=str, delimiter="\n")
print(labels)
nms.stringClassLabels.vector.extend(labels)
print(type(nms.stringClassLabels))
nms_model = coremltools.models.MLModel(nms_spec)
nms_model.save("NMS.mlmodel")
print(nms_model._spec)