import argparse
import os
import glob
import PIL
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

embedded_fingerprints_path = '/scratch2/users/carl/Stegastamp/Parallel_AF_coco/fingerprint_images/embedded_fingerprints.txt'
detected_fingerprints_path = '/scratch2/users/carl/Stegastamp/Parallel_AF_coco/coco_instruct/detected_fingerprints.txt'

columns = ['filename', 'messages']
ans_message = pd.read_csv(embedded_fingerprints_path, header=None, names=columns)
detect_message = pd.read_csv(detected_fingerprints_path, header=None, names=columns)

decoding_accuracy = 0.0
for i in tqdm(range(len(detect_message))):
    answer_message = ans_message['messages'][i]
    # import pdb; pdb.set_trace()
    # detection_message = (detect_message.loc[ans_message['filename'][i] == detect_message['filename']]).values[0][-1]
    detection_message = detect_message['messages'][i]
    answer_message = [int(i) for i in answer_message]
    detection_message = [int(i) for i in detection_message]
    answer_message = torch.tensor(answer_message)
    detection_message = torch.tensor(detection_message)

    bitwise_avg_err = ((np.abs(detection_message - answer_message)).sum() / detection_message.shape[0]).item()
    decoding_accuracy += bitwise_avg_err

# import pdb;pdb.set_trace()
decoded_accuracy = (1.0-decoding_accuracy/len(ans_message)) * 100
print("decoded_accuracy=", decoded_accuracy)