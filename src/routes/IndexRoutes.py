from flask import Blueprint, render_template, jsonify, redirect, url_for, Response
# Services
from src.services.Face_detector import RecognitionEmotionService

# Logger
import traceback
from src.utils.Logger import Logger


main = Blueprint('index_blueprint', __name__)


@main.route('/')
def login():
    try:
        return render_template("index.html")
    except Exception as ex:
        print("error: " + str(ex))

@main.route('/dashboard')
def dashboard():
    try:
        return render_template("dashboard3.html")
    except Exception as ex:
        print("error: " + str(ex))

# @main.route('/dashboard2')
# def dashboard2():
#     try:
#         return render_template("dashboard2.html")
#     except Exception as ex:
#         print("error: " + str(ex))

# @main.route('/record_class')
# def record_class():
#     try:
#         return render_template("recordClassItem.html")
#     except Exception as ex:
#         print("error: " + str(ex))

# @main.route('/video_stream', methods=['GET'])
# def video_stream():
#     try:
#         recognition_emotion_service = RecognitionEmotionService()
#         video_stream = recognition_emotion_service.video_stream()
#         # return jsonify(video_stream)   
    
#         return Response(video_stream, mimetype='multipart/x-mixed-replace; boundary=frame')
#     except Exception as ex:
#         Logger.add_to_log("error", str(ex))
#         Logger.add_to_log("error", traceback.format_exc())
#         return jsonify({'message': "ERROR", 'success': False})
 

@main.route('/video_stream_live', methods=['GET'])
def video_stream_live():
    try:
        recognition_emotion_service = RecognitionEmotionService()
        video_stream = recognition_emotion_service.video_stream_online()
        
        return Response(video_stream, mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as ex:
        Logger.add_to_log("error", str(ex))
        Logger.add_to_log("error", traceback.format_exc())
        return jsonify({'message': "ERROR", 'success': False})

