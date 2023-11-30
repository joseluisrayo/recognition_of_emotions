#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import numpy as np
import tensorflow as tf
import cv2
import os

from utils import label_map_util
from utils import visualization_utils_color as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/mobilenet_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'
PATH_TO_LABELS2 = './protos/face_label_mapEmotion.pbtxt'

NUM_CLASSES = 2
NUM_CLASSES_EMOT = 5

# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(
#     label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# category_index = label_map_util.create_category_index(categories)

#EMOTION
# label_map2 = label_map_util.load_labelmap(PATH_TO_LABELS2)
# categories = label_map_util.convert_label_map_to_categories(
#     label_map2, max_num_classes=NUM_CLASSES_EMOT, use_display_name=True)

categories = [
    {'id': 0, 'name': 'Neutral'},
    {'id': 1, 'name': 'Happy'},
    {'id': 2, 'name': 'Sad'},
    {'id': 3, 'name': 'Surprise'},
    {'id': 4, 'name': 'Angry'},
]
category_index = label_map_util.create_category_index(categories)


# class TensoflowFaceDector(object):
#     def __init__(self, PATH_TO_CKPT):
#         """Tensorflow detector
#         """

#         self.detection_graph = tf.Graph()
#         with self.detection_graph.as_default():
#             od_graph_def = tf.compat.v1.GraphDef()
#             with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
#                 serialized_graph = fid.read()
#                 od_graph_def.ParseFromString(serialized_graph)
#                 tf.import_graph_def(od_graph_def, name='')

#         with self.detection_graph.as_default():
#             config = tf.compat.v1.ConfigProto()
#             config.gpu_options.allow_growth = True
#             self.sess = tf.compat.v1.Session(
#                 graph=self.detection_graph, config=config)
#             self.windowNotSet = True

#     def run(self, image):
#         """image: bgr image
#         return (boxes, scores, classes, num_detections)
#         """

#         image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # the array based representation of the image will be used later in order to prepare the
#         # result image with boxes and labels on it.
#         # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#         image_np_expanded = np.expand_dims(image_np, axis=0)
#         image_tensor = self.detection_graph.get_tensor_by_name(
#             'image_tensor:0')
#         # Each box represents a part of the image where a particular object was detected.
#         boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
#         # Each score represent how level of confidence for each of the objects.
#         # Score is shown on the result image, together with the class label.
#         scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
#         classes = self.detection_graph.get_tensor_by_name(
#             'detection_classes:0')
#         num_detections = self.detection_graph.get_tensor_by_name(
#             'num_detections:0')
#         # Actual detection.
#         start_time = time.time()
#         (boxes, scores, classes, num_detections) = self.sess.run(
#             [boxes, scores, classes, num_detections],
#             feed_dict={image_tensor: image_np_expanded})
#         elapsed_time = time.time() - start_time
#         print('inference time cost: {}'.format(elapsed_time))

#         return (boxes, scores, classes, num_detections)


# Leer mobilenet_graph.pb
with tf.io.gfile.GFile(PATH_TO_CKPT,'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as mobilenet:
    tf.import_graph_def(graph_def,name='')


def load_image(image):
    return  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def detect_faces(image, score_threshold=0.7):
    global boxes, scores
    (imh, imw) = image.shape[:-1]
    img = np.expand_dims(image,axis=0)

    # Inicializar mobilenet
    sess = tf.compat.v1.Session(graph=mobilenet)
    image_tensor = mobilenet.get_tensor_by_name('image_tensor:0')
    boxes = mobilenet.get_tensor_by_name('detection_boxes:0')
    scores = mobilenet.get_tensor_by_name('detection_scores:0')

    # Predicción (detección)
    (boxes, scores) = sess.run([boxes, scores], feed_dict={image_tensor:img})

    # Reajustar tamaños boxes, scores
    boxes = np.squeeze(boxes,axis=0)
    scores = np.squeeze(scores,axis=0)

    # Depurar bounding boxes
    idx = np.where(scores>=score_threshold)[0]

    # Crear bounding boxes
    bboxes = []
    for index in idx:
        ymin, xmin, ymax, xmax = boxes[index,:]
        (left, right, top, bottom) = (xmin*imw, xmax*imw, ymin*imh, ymax*imh)
        left, right, top, bottom = int(left), int(right), int(top), int(bottom)
        bboxes.append([left,right,top,bottom])

    return bboxes

def draw_box(image,box,color,line_width=6):
    if box==[]:
        return image
    else:
        cv2.rectangle(image,(box[0],box[2]),(box[1],box[3]),color,line_width)
    return image

def extract_faces(image,bboxes,new_size=(160,160)):
    cropped_faces = []
    for box in bboxes:
        left, right, top, bottom = box
        face = image[top:bottom,left:right]
        cropped_faces.append(cv2.resize(face,dsize=new_size))
    return cropped_faces

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage:%s (cameraID | filename) Detect faces in the video example:%s 0" % (
            sys.argv[0], sys.argv[0]))
        exit(1)

    try:
        camID = int(sys.argv[1])
    except:
        camID = sys.argv[1]

    #tDetector = TensoflowFaceDector(PATH_TO_CKPT)

    cap = cv2.VideoCapture(camID)
    windowNotSet = True

    model = tf.keras.models.load_model('model/expression.model')
    labels = ["Neutral","Happy","Sad","Surprise","Angry"]

    while True:
        ret, image = cap.read()
        if ret == 0:
            break

        images = load_image(image)
        bboxes = detect_faces(images)
        face = extract_faces(images,bboxes)

        #add dimensiones
        for faces in face:
            # Convertir a escala de grises
            gray_image = cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY)

            # Redimensionar a (48, 48)
            resized_image = cv2.resize(gray_image, (48, 48))

            # Expandir las dimensiones si es necesario
            resized_image_expanded = np.expand_dims(resized_image, axis=0)

            predictions = model.predict(resized_image_expanded).argmax()

            print(labels[predictions])

        [h, w] = image.shape[:2]
        image = cv2.flip(image, 1)

        # (boxes, scores, classes, num_detections) = tDetector.run(image)

        # Convertir a escala de grises para preduccion
        # gray_image = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # resized_image = cv2.resize(gray_image, (48, 48))
        # resized_image_expanded = np.expand_dims(resized_image, axis=0)

        # # predecimos la emocion de la imagen
        # predictions = model.predict(face).argmax()
        # print(predictions)
        # #convertir a numpy la prediccion
        # classes_array_resultante = np.ones((1, 100)) * predictions

        # vis_util.visualize_boxes_and_labels_on_image_array(
        #     image,
        #     np.squeeze(boxes),
        #     np.squeeze(classes_array_resultante).astype(np.int32), #classes []
        #     np.squeeze(scores),
        #     category_index,
        #     use_normalized_coordinates=True,
        #     line_thickness=4)

        # if windowNotSet is True:
        #     cv2.namedWindow("tensorflow based (%d, %d)" %
        #                     (w, h), cv2.WINDOW_NORMAL)
        #     windowNotSet = False

        cv2.imshow("tensorflow based (%d, %d)" % (620, 360), image)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break

    cap.release()
