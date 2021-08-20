import os
from os.path import join
from utils_funcs import *
import glob2
import cv2
import imutils
import numpy as np
import requests

import flask
from flask import Flask
from flask import Response
from flask import request
from flask import send_from_directory

import json
import shutil
from datetime import datetime
import mimetypes
import warnings

import torch
import pandas as pd
import face_alignment

warnings.filterwarnings("ignore")

app = Flask(__name__)

UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.isdir(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

EYE_AR_THRESH = 0.3

# Initialize models
model = torch.hub.load('.', 'custom', path='headpose.pt', source='local')
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

# For orientation request
url = "http://192.168.1.27:8171/"
payload = {'debug': 'True'}
headers = {}

# Requests
@app.route('/')
def home():
    return flask.jsonify({"server": 1})

@app.route('/headpose', methods=['POST'])
def detect_headpose():
    sequence = request.form.get('sequence', '')
    video = request.files.get("video")

    # Process sequence
    sequence = list(sequence.upper())
    if not all([s in ['L', 'R', 'U', 'D'] for s in sequence]):
        return {"error": "invalid sequence"}
    
    # Save file
    if video.filename.strip() == '':
        return {"error": "file is not selected"}
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
    while True:
        ret, frame = cap.read()
        count += 1
        if ret:
            # Find orientation angle
            if count == 1:
                cv2.imwrite('to_check.jpg', frame)
                files = ('image',('to_check.jpg',open('to_check.jpg','rb'),'image/jpeg'))
                response = requests.request("POST", url, headers=headers, data=payload, files=files)
                response = json.loads(response.text)
                angle = response['angle']

            frame = imutils.rotate(frame, angle=-angle)

            result = model(frame[:, :, ::-1]).pandas().xyxy[0].sort_values(by=['confidence'])
            if len(result) == 0:
                continue

            detected_pose = label_map[result['class'][0]]

            # Check whether eyes are closed
            if detected_pose == 'F':
                preds = fa.get_landmarks(frame[:, :, ::-1])[0].astype(np.int32)
                average_ear = compute_EAR(preds)
                if average_ear < EYE_AR_THRESH:
                    raw_sequence.append('E')
                else:
                    raw_sequence.append('F')
            else:
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
            'down': len(frames['D'])
            }
    }
    return Response(json.dumps(response),  mimetype='application/json')

@app.route('/get-image', methods=['GET'])
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
    