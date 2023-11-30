import cv2
import json
import numpy as np
import os
import traceback
from src.utils.Logger import Logger

from src.utils import label_map_util
from src.utils import visualization_utils_color as vis_util
from .TensorflowDetector import TensoflowFaceDector

PATH_TO_CKPT = "src/model/mobilenet_graph.pb"
NUM_CLASSES = 2
PATH_TO_LABELS = "src/protos/face_label_map.pbtxt"

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


class RecognitionEmotionService():

    def video_stream_online(self):
        try:
            tDetector = TensoflowFaceDector(PATH_TO_CKPT)
            frame_size = (885, 472)

            cap = cv2.VideoCapture(0)
            windowNotSet = True
            while True:
                ret, image = cap.read()
                if ret == 0:
                    break

                [h, w] = image.shape[:2]
                print(h, w)
                image = cv2.flip(image, 1)

                (boxes, scores, classes, num_detections) = tDetector.run(image)

                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=4)

                # if windowNotSet is True:
                #     cv2.namedWindow("tensorflow based (%d, %d)" %
                #                     (w, h), cv2.WINDOW_NORMAL)
                #     windowNotSet = False

                # cv2.imshow("tensorflow based (%d, %d)" % (w, h), image)
                # k = cv2.waitKey(1) & 0xff
                # if k == ord('q') or k == 27:
                #     break

                encodedImage = cv2.imencode('.jpg', image)[1].tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encodedImage + b'\r\n')


        except Exception as ex:
            print("error: " + str(ex))
        finally:
            cap.release()

    # def video_stream(self):
    #     try:
    #         cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #         face_cascade_path = "src\haarcascade\haarcascade_frontalface_default.xml"
    #         face_detector = cv2.CascadeClassifier(face_cascade_path)

    #         # carpeta data
    #         dataPath = 'src\data'
    #         imagePaths = os.listdir(dataPath)

    #         # modelo pre entrenado
    #         method = 'LBPH'
    #         if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
    #         if method == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
    #         if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

    #         emotion_recognizer.read('src\modelo' + method + '.xml')

    #         while True:
    #             ret, frame = cap.read()
    #             if ret:
    #                 # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    #                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #                 faces = face_detector.detectMultiScale(gray, 1.3, 5)
    #                 auxFrame = gray.copy()

    #                 # agregamos un nuevo frame
    #                 nFrame = cv2.hconcat([frame, np.zeros((480,300,3), dtype=np.uint8)])

    #                 for (x, y, w, h) in faces:
    #                     rostro = auxFrame[y:y + h, x:x + w]
    #                     rostro = cv2.resize(rostro, (150, 150), interpolation = cv2.INTER_CUBIC)
    #                     result = emotion_recognizer.predict(rostro)

    #                     cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
    #                     if method == 'LBPH':
    #                         if result[1] < 60:
    #                             cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x,y-25) ,2, 1.1, (0,255,0), 1, cv2.LINE_AA)
    #                             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #                             image = self.emotionImage(imagePaths[result[0]])
    #                             nFrame = cv2.hconcat([frame,image])
    #                         else:
    #                             cv2.putText(frame ,'No identificado', (x,y-20), 2, 0.8, (0,0,255), 1, cv2.LINE_AA)
    #                             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #                             nFrame = cv2.hconcat([frame, np.zeros((480,300,3), dtype=np.uint8)])

    #                 encodedImage = cv2.imencode('.jpg', nFrame)[1].tobytes()
    #                 yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encodedImage + b'\r\n')

    #     except Exception as ex:
    #         Logger.add_to_log("error", str(ex))
    #         Logger.add_to_log("error", traceback.format_exc())
    #     finally:
    #         cap.release()

    # def emotionImage(self, emotion):
    #     # Emojis
    #     if emotion == 'Felicidad': image = cv2.imread('src/static/img/Emojis/felicidad.jpeg')
    #     if emotion == 'Enojo': image = cv2.imread('src/static/img/Emojis/enojo.jpeg')
    #     if emotion == 'Sorpresa': image = cv2.imread('src/static/img/Emojis/sorpresa.jpeg')
    #     if emotion == 'Tristeza': image = cv2.imread('src/static/img/Emojis/tristeza.jpeg')
    #     return image

    # def video_stream_online(self):
    #     try:

    #         # Definir el tamaño deseado para el frame (ancho, alto)
    #         frame_size = (885, 472)

    #         cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #         cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
    #         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])

    #         face_cascade_path = "src\haarcascade\haarcascade_frontalface_default.xml"
    #         face_detector = cv2.CascadeClassifier(face_cascade_path)

    #         while True:
    #             ret, frame = cap.read()
    #             if ret:
    #                 frame = cv2.resize(frame, frame_size)

    #                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    #                 faces = face_detector.detectMultiScale(gray, 1.3, 5)
    #                 for (x, y, w, h) in faces:
    #                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #                 encodedImage = cv2.imencode('.jpg', frame)[1].tobytes()
    #                 yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encodedImage + b'\r\n')

    #     except Exception as ex:
    #         print("error: " + str(ex))
    #     finally:
    #         cap.release()
