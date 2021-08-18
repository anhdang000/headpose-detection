import os
from os.path import join
from utils_funcs import calc_score, filter_records
import glob2
import cv2

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

warnings.filterwarnings("ignore")

app = Flask(__name__)

UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.isdir(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

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
    
    if os.path.isdir('runs/detect'):
        shutil.rmtree('runs/detect')
        
    os.system(f'python detect.py --weights headpose.pt --source {file_path} --save-txt --save-conf')

    raw_sequence = []
    label_map = ['F', 'R', 'L', 'U', 'D']
    label_paths = glob2.glob('runs/detect/exp/labels/*txt')
    frame_ids = [int(path.split('/')[-1].split('_')[-1].split('.')[0]) for path in label_paths]
    count = 0
    frames = {'F': [], 'R': [], 'L': [], 'U': [], 'D': []}
    case_map = {'F': 'front', 'L': 'left', 'R': 'right', 'U': 'up', 'D': 'down'}
    frame = None
    diffs = []
    while True:
        prev = frame
        ret, frame = cap.read()
        count += 1
        if ret and count in frame_ids:
            with open(label_paths[frame_ids.index(count)], 'r') as f:
                label = [line.strip().split(' ') for line in  f.readlines()]
            label = sorted(label, key=lambda entry: entry[-1],  reverse=True)
            detected_pose = label_map[int(label[0][0])]
            raw_sequence.append(detected_pose)
            
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

            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, frame)
                diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                diff = cv2.GaussianBlur(diff, (5, 5), 0)
                diffs.append(abs(diff.mean()))
        else:
            break

    diffs = np.array(diffs)
    if (diffs > 3).sum() > 5:
        is_cheated = True
    else:
        is_cheated = False

    detected_sequence = filter_records(raw_sequence, patience=2)
    score = calc_score(detected_sequence, sequence)
    if detected_sequence == sequence:
        message = "success"
    else:
        message = "fail"
    response = {
        "message": message,
        "is_cheated": is_cheated,
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
    