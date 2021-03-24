# Load an image as input
import PIL.Image
import requests
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import cv2
import random


def get_color_table(class_num, seed=10):
    random.seed(seed)
    color_table = {}
    for i in range(class_num):
        color_table[i] = [random.randint(0, 255) for _ in range(3)]
    return color_table

## 加载文本中的类别 <class 'numpy.ndarray'>
labels = np.loadtxt("coco_labels.txt", dtype=str, delimiter="\n")

# 边框颜色集(等于类别数)
color_table = get_color_table(len(labels))

imgs_path = ['image/002.jpg']
for img_path in imgs_path:

	img_url = img_path
	img = PIL.Image.open(img_url)
	#plt.imshow(np.asarray(img))
	#plt.show()

	# Preprocess the image - normalize to [-1,1]
	img = img.resize([320,320], PIL.Image.ANTIALIAS)


	import coremltools
	# Input shape should be [1,1,3,300,300]
	mlmodel_path = 'ssd_mobilenet.mlmodel'

	mlmodel = coremltools.models.MLModel(mlmodel_path)

	# Pay attention to '__0'. We change ':0' to '__0' to make sure MLModel's 
	# generated Swift/Obj-C code is semantically correct
	coreml_input_name = "Preprocessor/sub"
	coreml_output_names = ["Postprocessor/raw_box_scores", "Postprocessor/raw_box_encodings"]
	coreml_input = {coreml_input_name: img}

	# When useCPUOnly == True, Relative error should be around 0.001
	# When useCPUOnly == False on GPU enabled devices, relative errors 
	# are expected to be larger due to utilization of lower-precision arithmetics

	coreml_outputs_dict = mlmodel.predict(coreml_input)
	coreml_outputs = [coreml_outputs_dict[out_name] for out_name in 
	                  coreml_output_names]
	## 四输出：1.原始bbox数组，2.置信数组，3.框id，4.框数
	coreml_scores, coreml_box_encodings = coreml_outputs

	print('*'*15 + '原始输出' + '*'*15)
	print('coreml_box_encodings.shape:', coreml_box_encodings.shape)
	print('coreml_scores.shape:', coreml_scores.shape)
	print('*'*38)

	##########################-- 输出结果预处理 --############################
	#-1. 去掉多余的维度（变成两维数组），但所得数组是经过padding的，加0补全到设定的maxNumbox数目
	twoD_boxes = np.squeeze(coreml_box_encodings)
	twoD_score = np.squeeze(coreml_scores)
	print('twoD_boxes.shape:', twoD_boxes.shape)
	print('twoD_score.shape:', twoD_score.shape)

	max_score_array = np.max(twoD_score, axis = 0)
	index = np.argmax(max_score_array)
	print(labels[index])
	for i in range(0, len(max_score_array)):
		print(max_score_array[i])
		print(i)
		print('*'*38)
