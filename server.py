import os
from os.path import join
from utils_funcs import *
import glob2
import cv2
import imutils
import numpy as np
import requests

import flask
from flask_cors import CORS, cross_origin
from flask import Flask
from flask import Response
from flask import request
from flask import send_file, send_from_directory

import json
import shutil
from datetime import datetime
import mimetypes
import warnings

import torch
import pandas as pd
# import face_alignment

warnings.filterwarnings("ignore")

app = Flask(__name__)
cors = CORS(app)

UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.isdir(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

EYE_AR_THRESH = 0.3

# Initialize models
model = torch.hub.load('.', 'custom', path='headpose.pt', source='local')
model_smile = torch.hub.load('.', 'custom', path='smile_facebox.pt', source='local')

# For orientation request
url = "http://192.168.1.27:8171/"
payload = {'debug': 'True'}
headers = {}

# Requests
@app.route('/')
@cross_origin()
def home():
    return flask.jsonify({"server": 1})

@app.route('/headpose', methods=['POST'])
@cross_origin()
def detect_headpose():
    sequence = request.form.get("sequence", '')
    video = request.files.get("video")
    unique_id = request.form.get("unique_id", '')

    # Process sequence
    sequence = list(sequence.upper())
    if not all([s in ['L', 'R', 'U', 'D', 'S'] for s in sequence]):
        return {"error": "invalid sequence"}
    
    # Save file
    if video is None:
        return {"error": "file is not selected"}
    elif video.filename.strip() == '':
        return {"error": "file is not selected"}
        
    if len(unique_id) > 0:
        file_id = unique_id + '_' + '_'.join(str(datetime.now()).split())
    else:
        file_id = '_'.join(str(datetime.now()).split())
        
    ext = video.filename.split('.')[-1]
    file_path = join(app.config['UPLOAD_FOLDER'], file_id + '.' + ext)
    video.save(file_path)

    # Check validation
    if not mimetypes.guess_type(file_path)[0].startswith('video'):
        return {"error": "invalid video"}
    
    # Init video reader
    try:
        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        os.mkdir(file_id)
    except:
        return {"error": "cannot read video"}

    raw_sequence = []
    label_map = ['F', 'R', 'L', 'U', 'D', 'S']
    count = 0
    frames = {'F': [], 'R': [], 'L': [], 'U': [], 'D': [], 'S': []}
    case_map = {'F': 'front', 'L': 'left', 'R': 'right', 'U': 'up', 'D': 'down', 'S': 'smile'}

    # Find orientation angle
    print('Finding orientation angle')
    cap_2 = cv2.VideoCapture(file_path)
    if total_frames > 0:
        frame_indices = range(1, total_frames, (total_frames - 1)//5)
    else:
        frame_indices = range(1, 6)
    for i in frame_indices:
        cap_2.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap_2.read()
        try:
            cv2.imwrite('to_check.jpg', frame)
        except:
            angle = None
            continue
        files = [('image',('to_check.jpg', open('to_check.jpg','rb'),'image/jpeg'))]
        response = requests.request("POST", url, headers=headers, data=payload, files=files)
        response = json.loads(response.text)
        angle = response['angle']
        if angle is not None:
            break
            
    if angle is None:
        return {"error_code": 2, "error": "invalid orientation"}


    print('Predicting frame by frame')
    while True:
        ret, frame = cap.read()
        count += 1
        if ret:
            frame = imutils.rotate(frame, angle=-angle)

            result = model(frame[:, :, ::-1]).pandas().xyxy[0].sort_values(by=['confidence'], ascending=False)
            if len(result) == 0:
                continue

            detected_pose = label_map[result['class'][0]]

            # Run smile detection model
            if detected_pose == 'F':
                smile_result = model_smile(frame[:, :, ::-1]).pandas().xyxy[0].sort_values(by=['confidence'])
                if len(smile_result) > 0:
                    detected_pose = 'S'
            raw_sequence.append(detected_pose)

            # Append normal results and save into images 
            if count < 3 or count > total_frames - 3:
                frames['F'].append(frame)
                if not os.path.isdir(join(file_id, 'front')):
                    os.makedirs(join(file_id, 'front'))
                curr_num_imgs = len(frames['F']) - 1
                cv2.imwrite(join(file_id, 'front', f'{curr_num_imgs}.jpg'), frame)
            else:
                frames[detected_pose].append(frame)
                case = case_map[detected_pose]
                if not os.path.isdir(join(file_id, case)):
                    os.makedirs(join(file_id, case))
                curr_num_imgs = len(frames[detected_pose]) - 1
                cv2.imwrite(join(file_id, case, f'{curr_num_imgs}.jpg'), frame)
        else:
            break

    detected_sequence = filter_records(raw_sequence, patience=2)
    score = calc_score(detected_sequence, sequence)
    if detected_sequence == sequence:
        message = "success"
    else:
        message = "fail"
    response = {
        "message": message,
        "score": score,
        "detected_sequence": '-'.join(detected_sequence),
        "raw_sequence": '-'.join(raw_sequence),
        "folder": file_id,
        "num_images": {
            'front': len(frames['F']), 
            'right': len(frames['R']), 
            'left': len(frames['L']),
            'up': len(frames['U']),
            'down': len(frames['D']),
            'smile': len(frames['S'])
            }
    }
    return Response(json.dumps(response),  mimetype='application/json')

@app.route('/get-image', methods=['GET'])
@cross_origin()
def get_image():
    folder = request.args.get('folder')
    case = request.args.get('case')
    id = request.args.get('id')

    if not os.path.exists(join(folder, case, id)):
        return {"error": "file_not_found"}
    else:
        return send_from_directory(
            join(folder, case), 
            id, 
            as_attachment=True
            )
    
@app.route('/get-video', methods=['GET'])
@cross_origin()
def get_video():
    file_id = request.args.get('file_id')
    file_path = glob2.glob(join(app.config['UPLOAD_FOLDER'], file_id + '*'))[0]
    return send_file(file_path)