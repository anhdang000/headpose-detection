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

import pandas as pd

from configs import *


app = Flask(__name__)
cors = CORS(app)

UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.isdir(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

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
    images = request.files.getlist("images")
    unique_id = request.form.get("unique_id", '')

    # Remove invalid image(s)
    images = [image for image in images if image.filename != '']

    # Find source
    if video is not None and video.filename != '' and len(images) > 0:
        return {"code": 4, "message": "too many sources"}
    elif video is not None and video.filename != '':
        source = 'video'
    elif len(images) > 0:
        source = 'images'
    else:
        source = None

    # Process sequence
    sequence = list(sequence.upper())
    if not all([s in ['L', 'R', 'U', 'D', 'S'] for s in sequence]):
        return {"error": "invalid sequence"}
    
    if len(unique_id) > 0:
        file_id = unique_id + '_' + '_'.join(str(datetime.now()).split())
    else:
        file_id = '_'.join(str(datetime.now()).split())
        
    if source is None:
        return {"code": 5, "message": "no source found"}

    ########################################################################### 
    ### Video
    ###########################################################################
    elif source == 'video':
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
        except:
            return {"error": "cannot read video"}

        # Find orientation angle
        logging.info(f'Video length {total_frames}')
        cap_2 = cv2.VideoCapture(file_path)
        if total_frames > 0:
            frame_indices = range(1, total_frames, (total_frames - 1)//5)
        else:
            frame_indices = range(1, 6)

        logging.info('Finding orientation angle')
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
        
        logging.info(f'Orientation: {angle}')
        if angle is None:
            return {"error_code": 2, "error": "invalid orientation"}

        raw_sequence = []
        frames = {'F': [], 'R': [], 'L': [], 'U': [], 'D': [], 'S': []}
        count = 0
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
        logging.info(f'Detected sequence: {detected_sequence}\tScore: {score}')
        return Response(json.dumps(response),  mimetype='application/json')

    ########################################################################### 
    ### Images
    ###########################################################################
    elif source == 'images':
        folder_path = join(app.config['UPLOAD_FOLDER'], file_id)
        os.mkdir(folder_path)
        imgs = []
        angle = None
        logging.info('Finding orientation angle')
        for i in range(len(images)):
            image_path = join(folder_path, images[i].filename)
            images[i].save(image_path)
            try:
                img = cv2.imread(image_path)
                img = img[:, :, ::-1]
            except:
                logging.warning(f'Invalid file: {images[i].filename}')
                continue
            imgs.append(img)

            files = [('image',(image_path, open(image_path,'rb'),'image/jpeg'))]
            response = requests.request("POST", url, headers=headers, data=payload, files=files)
            response = json.loads(response.text)
            result_angle = response['angle']
            if result_angle is not None:
                angle = result_angle

        logging.info(f'Orientation: {angle}')
        if len(imgs) == 0:
            return {"code": 6, "message": "invalid image"}
        elif angle is None:
            return {"error_code": 2, "error": "invalid orientation"}

        raw_sequence = []
        frames = {'F': [], 'R': [], 'L': [], 'U': [], 'D': [], 'S': []}
        for i in range(len(images)):
            img = imutils.rotate(imgs[i], angle=-angle)

            # Run standard detection model
            result = model(img).pandas().xyxy[0].sort_values(by=['confidence'], ascending=False)
            if len(result) == 0:
                continue
            detected_pose = label_map[result['class'][0]]

            # Run smile detection model
            if detected_pose == 'F':
                smile_result = model_smile(img).pandas().xyxy[0].sort_values(by=['confidence'])
                if len(smile_result) > 0:
                    detected_pose = 'S'
            raw_sequence.append(detected_pose)

            # Append normal results and save into images 
            frames[detected_pose].append(img)
            case = case_map[detected_pose]
            if not os.path.isdir(join(file_id, case)):
                os.makedirs(join(file_id, case))
            curr_num_imgs = len(frames[detected_pose]) - 1
            cv2.imwrite(join(file_id, case, f'{curr_num_imgs}.jpg'), img[:, :, ::-1])

        detected_sequence = filter_records(raw_sequence, patience=1)
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
        logging.info(f'Detected sequence: {detected_sequence}\tScore: {score}')
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
