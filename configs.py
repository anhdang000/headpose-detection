import sys
import torch
import logging
import warnings

# Warning 
warnings.filterwarnings("ignore")

# Config logger
logging.basicConfig(
    level=logging.INFO,
    format = (
        '%(levelname)s:\t'
        '%(filename)s:'
        '%(funcName)s():'
        '%(lineno)d\t'
        '%(message)s'
        )
    )

# Config orientation request
url = "http://192.168.1.27:8171/single_image"
payload = {'debug': 'False'}
headers = {}

label_map = ['F', 'R', 'L', 'U', 'D', 'S']
case_map = {'F': 'front', 'L': 'left', 'R': 'right', 'U': 'up', 'D': 'down', 'S': 'smile'}

# Initialize models
model = torch.hub.load('.', 'custom', path='headpose.pt', source='local')
model_smile = torch.hub.load('.', 'custom', path='smile_facebox.pt', source='local')