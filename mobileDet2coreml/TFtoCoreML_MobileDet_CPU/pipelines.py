# ===============================================
# PART 4: Putting it all together into a pipeline
# ===============================================
import coremltools
import numpy as np
from coremltools.models.pipeline import *
from coremltools.models import datatypes
from coremltools.models import neural_network

# The number of predicted classes, excluding background.
num_classes = 90

# The number of predicted bounding boxes.
num_anchors = 2034

# Size of the expected input image.
input_width = 320
input_height = 320

default_iou_threshold = 0.4
default_confidence_threshold = 0.5

ssd_path = 'ssd_mobileDet.mlmodel'
ssd_model = coremltools.models.MLModel(ssd_path)

decoder_path = 'Decoder.mlmodel'
decoder_model = coremltools.models.MLModel(decoder_path)

nms_path = 'NMS.mlmodel'
nms_model = coremltools.models.MLModel(nms_path)



input_features = [ ("Preprocessor/sub", datatypes.Array(3, 320, 320)),
                   ("iouThreshold", datatypes.Double()),
                   ("confidenceThreshold", datatypes.Double())]

output_features = [ "confidence", "coordinates"]

pipeline = Pipeline(input_features, output_features)

# We added a dimension of size 1 to the back of the inputs of the decoder 
# model, so we should also add this to the output of the SSD model or else 
# the inputs and outputs do not match and the pipeline is not valid.
ssd_output = ssd_model._spec.description.output
ssd_output[0].type.multiArrayType.shape[:] = [1, num_anchors, num_classes + 1]
ssd_output[1].type.multiArrayType.shape[:] = [1, num_anchors, 4]

pipeline.add_model(ssd_model)
pipeline.add_model(decoder_model)
pipeline.add_model(nms_model)

# The "image" input should really be an image, not a multi-array.
pipeline.spec.description.input[0].ParseFromString(ssd_model._spec.description.input[0].SerializeToString())

# Copy the declarations of the "confidence" and "coordinates" outputs.
# The Pipeline makes these strings by default.
pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())
pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())

# Add descriptions to the inputs and outputs.
pipeline.spec.description.input[1].shortDescription = "(optional) Radius of suppression"
pipeline.spec.description.input[2].shortDescription = "(optional) Remove bounding boxes below this threshold"
pipeline.spec.description.output[0].shortDescription = u"Boxes \xd7 Class confidence"
pipeline.spec.description.output[1].shortDescription = u"Boxes \xd7 [x, y, width, height] (relative to image size)"

# Add metadata to the model.
pipeline.spec.description.metadata.versionString = "MobileDet_SSDLite"
pipeline.spec.description.metadata.shortDescription = "MobileDet + SSDLite(tensorflow), trained on COCO"
pipeline.spec.description.metadata.author = "ZHANG KUN"
pipeline.spec.description.metadata.license = "ryukyus"

# Add the list of class labels and the default threshold values too.
labels = np.loadtxt("coco_labels.txt", dtype=str, delimiter="\n")
user_defined_metadata = {
    "iou_threshold": str(default_iou_threshold),
    "confidence_threshold": str(default_confidence_threshold),
    "classes": ",".join(labels)
}
pipeline.spec.description.metadata.userDefined.update(user_defined_metadata)

# Don't forget this or Core ML might attempt to run the model on an unsupported
# operating system version!（3及以上）
pipeline.spec.specificationVersion = 4

final_coreml_model_path = 'MobileDet_SSDLite.mlmodel'
final_model = coremltools.models.MLModel(pipeline.spec)
final_model.save(final_coreml_model_path)

print(final_model)
print("Done!")