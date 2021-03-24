# Load an image as input
import PIL.Image
import requests
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import cv2
import random
import time

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

imgs_path = ['image/bagong.jpg', 'image/messi.jpg', 'image/bagong1.jpg', 'image/bagong2.jpg', 'image/cat.jpg', 'image/person_cat.jpg', 'image/person_dog.jpg', 'image/person.jpg']
for img_path in imgs_path:

	img_url = img_path
	img = PIL.Image.open(img_url)
	#plt.imshow(np.asarray(img))
	#plt.show()

	# 'Image' object
	img = img.resize([320,320], PIL.Image.ANTIALIAS)
	#print("img: ", img.shape)

	import coremltools
	# Input shape should be [1,1,3,300,300]
	mlmodel_path = 'MobileNetV3_SSDLite_Large.mlmodel'

	mlmodel = coremltools.models.MLModel(mlmodel_path)

	# Pay attention to '__0'. We change ':0' to '__0' to make sure MLModel's 
	# generated Swift/Obj-C code is semantically correct
	coreml_input_name = "Preprocessor/sub"
	coreml_output_names = ["confidence", "coordinates"]
	coreml_input = {coreml_input_name: img}

	# When useCPUOnly == True, Relative error should be around 0.001
	# When useCPUOnly == False on GPU enabled devices, relative errors 
	# are expected to be larger due to utilization of lower-precision arithmetics
	start_time = time.time()
	coreml_outputs_dict = mlmodel.predict(coreml_input, useCPUOnly=True)
	end_time = time.time()

	coreml_outputs = [coreml_outputs_dict[out_name] for out_name in 
	                  coreml_output_names]
	## 四输出：1.原始bbox数组，2.置信数组，3.框id，4.框数
	coreml_scores, coreml_box_encodings = coreml_outputs

	#print('*'*15 + '原始输出' + '*'*15)
	print('coreml_box_encodings.shape:', coreml_box_encodings.shape)
	print('coreml_scores.shape:', coreml_scores.shape)
	print('*'*38)

	##########################-- 输出结果预处理 --#############################
	
	##置信最高索引类别 <class 'numpy.ndarray'>
	max_score_array = coreml_scores.argmax(axis = 1)
	#print(max_score_array)
	confidences = []
	classes_labels = []

	#-4. 获取类别和置信列表
	for i in range(0, coreml_scores.shape[0]):
		index = max_score_array[i]
		classes_labels.append(labels[index])
		confidences.append(coreml_scores[i][index])
		
	print('*'*15 + '处理后输出' + '*'*15)
	print('目标类别索引: ', max_score_array)
	print('目标置信列表: ', confidences)
	print('目标列别标签列表: ', classes_labels)
	print('*'*38)


	image_original = cv2.imread(img_url)
	# 获得元输入图片宽，高
	height_ori, width_ori = image_original.shape[:2]

	##-- 处理归一化的bbox（x_center*width_ori, y_center*height_ori, W*width_ori, H*height_ori） --##
	coreml_box_encodings[:,0] = width_ori*coreml_box_encodings[:,0]
	coreml_box_encodings[:,1] = height_ori*coreml_box_encodings[:,1]
	coreml_box_encodings[:,2] = width_ori*coreml_box_encodings[:,2]
	coreml_box_encodings[:,3] = height_ori*coreml_box_encodings[:,3]
	print('true_norm_box: ', coreml_box_encodings)

	#############################-- opencv 绘制矩形框--#####################################
	for i in range(0, coreml_scores.shape[0]):
		## opencv图片上画矩形，需转化为（(xmin,ymin）,(xmax,ymax))形式
		xmin = int(coreml_box_encodings[i,0] - coreml_box_encodings[i,2]/2)
		ymin = int(coreml_box_encodings[i,1] - coreml_box_encodings[i,3]/2)
		xmax = int(coreml_box_encodings[i,0] + coreml_box_encodings[i,2]/2)
		ymax = int(coreml_box_encodings[i,1] + coreml_box_encodings[i,3]/2)

		color_index = max_score_array[i]
		#(图片，左上角坐标，右下角坐标，颜色，线条粗细)
		cv2.rectangle(image_original, (xmin, ymin), (xmax, ymax), color_table[color_index], 2)
		label = classes_labels[i] + ', {:.2f}%'.format(confidences[i] * 100)
		timeCost = '{:.2f}ms'.format((end_time - start_time)*1000)
		#(图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细, 线型)
		size = int(round(0.002 * max(image_original.shape[0:2])))
		cv2.putText(image_original, label, (xmin, ymin + 18), 0, float(size) / 3, color_table[color_index], thickness=size, lineType=cv2.LINE_AA)
		cv2.putText(image_original, timeCost, (0, 0 + 18), 0, float(size) / 3,(0, 0, 0), thickness=size, lineType=cv2.LINE_AA)
	cv2.imshow('Detection result', image_original)
	#cv2.imwrite('2.jpg', image_original)
	cv2.waitKey(0)
