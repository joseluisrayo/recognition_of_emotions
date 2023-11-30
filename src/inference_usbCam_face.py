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
# PATH_TO_LABELS = './protos/face_label_map.pbtxt'
PATH_TO_LABELS2 = './protos/face_label_mapEmotion.pbtxt'

# NUM_CLASSES = 2
NUM_CLASSES_EMOT = 5

# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(
#     label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# category_index = label_map_util.create_category_index(categories)

#EMOTION
label_map2 = label_map_util.load_labelmap(PATH_TO_LABELS2)
categories = label_map_util.convert_label_map_to_categories(
    label_map2, max_num_classes=NUM_CLASSES_EMOT, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


class TensoflowFaceDector(object):
    def __init__(self, PATH_TO_CKPT):
        """Tensorflow detector
        """

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.detection_graph.as_default():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(
                graph=self.detection_graph, config=config)
            self.windowNotSet = True

    def run(self, image):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name(
            'detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name(
            'num_detections:0')
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time
        print('inference time cost: {}'.format(elapsed_time))

        return (boxes, scores, classes, num_detections)


def detect_faces(imh, imw, boxes, scores, score_threshold=0.7):
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

def extract_faces(image,bboxes, new_size=(160,160)):
    cropped_faces = []
    for box in bboxes:
        left, right, top, bottom = box
        face = image[top:bottom, left:right]
        resized_face = cv2.resize(face, dsize=new_size)
        cropped_faces.append(resized_face)

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

    tDetector = TensoflowFaceDector(PATH_TO_CKPT)

    cap = cv2.VideoCapture(camID)
    windowNotSet = True

    model = tf.keras.models.load_model('model/expression.model')
    index = 0
    while True:
        ret, image = cap.read()
        if ret == 0:
            break

        # [imh, imw] = image.shape[:-1]
        [h, w] = image.shape[:2]
        image = cv2.flip(image, 1)

        (boxes, scores, classes, num_detections) = tDetector.run(image)

        bboxes = detect_faces(h, w, boxes, scores)
        faces = extract_faces(image,bboxes)
        
        print(faces)


        # boxees = np.squeeze(boxes)

        # STANDARD_COLORS = [
        #     'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
        #     'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
        #     'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
        #     'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
        #     'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
        #     'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
        #     'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
        #     'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
        #     'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
        #     'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
        #     'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
        #     'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
        #     'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
        #     'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
        #     'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
        #     'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
        #     'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
        #     'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
        #     'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
        #     'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
        #     'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
        #     'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
        #     'WhiteSmoke', 'Yellow', 'YellowGreen'
        # ]

        # import PIL.Image as Image
        # import PIL.ImageColor as ImageColor
        # import PIL.ImageDraw as ImageDraw
        # import PIL.ImageFont as ImageFont

        # bboxes = []
        # for idx, box in enumerate(boxees):
        #     color = 'Red'
        #     ymin, xmin, ymax, xmax = box
        #     bboxes.append([left,right,top,bottom])

            # image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
            # draw = ImageDraw.Draw(image_pil)
            # im_width, im_height = image_pil.size
            # (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
            #                               ymin * im_height, ymax * im_height)

            # # Dibuja el rectángulo en la imagen
            # draw.rectangle([left, top, right, bottom], outline=color, width=2)

            # # Guarda la imagen con el rectángulo en la carpeta de salida
            # output_path = os.path.join("C:\\Users\\51954\\Desktop\\ReconocimientoEmocionalWeb3\\src\\videos\\", f"image_with_box_{idx}.png")
            # image_pil.save(output_path)




        # # Convertir a escala de grises para preduccion
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # resized_image = cv2.resize(gray_image, (48, 48))
        # resized_image_expanded = np.expand_dims(resized_image, axis=0)

        # # predecimos la emocion de la imagen
        # predictions = model.predict(face).argmax()
        # print(predictions)
        # #convertir a numpy la prediccion
        # classes_array_resultante = np.ones((1, 100)) * predictions

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32), #classes []
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=4)

        if windowNotSet is True:
            cv2.namedWindow("tensorflow based (%d, %d)" %
                            (w, h), cv2.WINDOW_NORMAL)
            windowNotSet = False

        cv2.imshow("tensorflow based (%d, %d)" % (w, h), image)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break

    cap.release()
