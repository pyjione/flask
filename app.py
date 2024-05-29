from flask import Flask, request, jsonify
from fer import FER
import cv2
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import traceback
import os
import json
# Flask 애플리케이션 생성 코드
app = Flask(__name__)

# 현재 스크립트 파일의 절대 경로를 가져옴
script_dir = os.path.dirname(os.path.abspath(__file__))

# 로그 기록을 위한 코드
#formatter : 로그 메시지의 형식을 지정한다.
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#log_file : 로그 파일의 경로를 지정한다.
log_file = os.path.join(script_dir, 'app.log')
#file_handler는 로그 파일을 처리하기 위한 RotatingFileHandler를 설정한다. 
#이는 로그 파일이 일정 크기를 초과하면 새 파일로 백업된다.
file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=10)  # Save up to 10 MB of log files
file_handler.setLevel(logging.DEBUG)  # Set log level to DEBUG
file_handler.setFormatter(formatter)
app.logger.addHandler(file_handler)
# 애플리케이션에서 발생하는 모든 예외를 처리하는 함수이다. 
#예외가 발생하면 해당 예외 정보를 로그에 기록하고 클라이언트에게 오류 메시지를 JSON형식으로 반환한다.
@app.errorhandler(Exception)
def handle_error(e):
    # Print the traceback for detailed error information
    traceback.print_exc()

    # Log the error message
    app.logger.error(f"An error occurred: {str(e)}")

    # Return a JSON response with an error message
    return jsonify({'error': f'An error occurred: {str(e)}'}), 500
#Post메서드로 '/auth/detect-emotions'엔드포인틀르 생성한다. 
#클라이언트는 이미지를 업로드하고 이 엔드포인트로 POST요청을 보낸다.
@app.route('/auth/detect-emotions', methods=['POST'])
#이미지에서 감정을 감지하는 함수
def detect_emotions():
    try:
        # Print the request data for debugging
        app.logger.info("Request data: %s", request.files)

        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        #클라이언트로부터 전송된 파일을 읽음
        file = request.files['file']  # Accessing the file using the key 'file'

        # Check if the file is empty
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        #OpenCV를 사용하여 이미지를 디코딩함.
        #클라이언트가 업로드한 이미지 파일을 읽어들임. 'file'은 Flask의 request 객체에서 전달된 파일 객체
        #'read()'메서드를 사용하여 파일의 내용을 읽어들여 'img_stream'변수에 저장함
        img_stream = file.read()
        #'img_stream'에서 읽어들인 이미지 데이터를 넘파이 배열로 변환한다.
        #넘파이의 'frombuffer()'함수를 사용하여 바이트 데이터를 넘파이 배열로 변환
        #'np.unit8'은 배열의 데이터 타입을 8비트 부호 없는 정수로 지정
        nparr = np.frombuffer(img_stream, np.uint8)
        #OpenCV의 'imdecode()'함수를 사용하여 넘파이 배열로 변환된 이미지 데이터를 디코딩하여 이미지로 변환
        #이때, 이미지 데이터의형식은 넘파이 배열로 저장되어 있으므로 'imdecode()'함수를 사용하여 디코딩
        #cv2.IMREAD_COLOR플래그는 이미지를 컬러 이미지로 읽어들이도록 저장
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Initialize the FER detector
        detector = FER()

        # Detect emotions in the image
        result = detector.detect_emotions(img)

        # Serialize ndarray objects
        def serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Serialize result
        result_serialized = json.dumps(result, default=serialize)

        # Return the serialized result as JSON
        return jsonify(json.loads(result_serialized))
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    # Create the log file if it doesn't exist
    if not os.path.exists(log_file):
        open(log_file, 'w').close()
    
    app.run(debug=True, port=5001)
