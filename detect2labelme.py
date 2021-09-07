import torch
from PIL import Image
import numpy as np
import json
import io
import codecs
from tqdm import tqdm
import glob2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir')
args = parser.parse_args()

def encodeImageForJson(image):
    img_pil = Image.fromarray(image, mode='RGB')
    f = io.BytesIO()
    img_pil.save(f, format='PNG')
    data = f.getvalue()
    encData = codecs.encode(data, 'base64').decode()
    encData = encData.replace('\n', '')
    return encData

model = torch.hub.load('.', 'custom', path='headpose.pt', source='local')

image_paths = glob2.glob(join(args.data_dir, '*.jpg'))
image_paths.sort()
for image_path in tqdm(image_paths, total=len(image_paths)):
    image_id = image_path.split('/')[-1]
    labelfile = image_id[:-4] + '.json'

    labelme = {
        "version": "4.5.7", 
        "flags": {}, 
        "shapes": [], 
        "imagePath": None, 
        "imageData": None,
        "imageHeight": None,
        "imageWeight": None
        }

    image = np.array(Image.open(image_path))
    h, w = image.shape[:2]

    # Assign image infos
    labelme["imagePath"] = image_id
    labelme["imageData"] = encodeImageForJson(image)
    labelme["imageHeight"] = h
    labelme["imageWidth"] = w

    # Predict and assign to `shapes`
    results = model(image).pandas().xyxy[0]
    for idx, result in results.iterrows():
        label_entry = {
            "label": None, 
            "points": [[None, None], [None, None]], 
            "group_id": None, 
            "shape_type": "rectangle", 
            "flags": {}
            }
        
        label_entry["label"] = result["name"]
        label_entry["points"][0] = [result["xmin"], result["ymin"]]
        label_entry["points"][1] = [result["xmax"], result["ymax"]]
        
        labelme['shapes'].append(label_entry)

    # Write to json file
    with open(join(args.data_dir, labelfile), 'w') as f:
        json.dump(labelme, f)