# TFtoCoreML
Transform the [object detection model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) trained on TensorFlow to iOS [CoreML model](https://developer.apple.com/machine-learning/models/) type
## Folder List
* mlmodels_IOU0.4_Conf0.6  
Core ML models(iOS) transformed from the models that trained on tensorflow(`threshold: IOU=0.4; Confidece=0.6`)
* MobileNetV1_SSD/MobileNetV2_SSDLite/MobileNetV3_Large_SSDLite/MobileNetV3_Small_SSDLite/MobileDet_SSD_CPU
## Transform Process
![](https://github.com/popCain/TFtoCoreML/blob/main/image/tf2coreml_process.png)
> **The transform process from file list**
1. **check_modelNodes.py**  
Load the `frozen_inference_graph.pb`,and print the name of nodes in each layer. Then get the input_node_name(mobilenet) and output_node_name(scores/boundingboxes)
2. **frozenToSimplePB.py**    
Simplify the model(Strip unused subgraphs get simplified frozen graph-`ssd_mobilenet.pb`)
3. **simplePB_ToCoreml.py**  
Main network tranceform(`ssd_mobilenet.mlmodel`)
4. **Decoder.py**
____
![](https://github.com/popCain/TFtoCoreML/blob/main/image/decode_process.png)
____
6. **NMS.py**
7. **pipelines.py**
