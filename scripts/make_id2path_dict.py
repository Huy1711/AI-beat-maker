"""
This script create a txt file that in a dictionary format
        {<file_id>: <file_path>}
with <file_id> is the basename of the audio file,
and <file_path> is the path to that file.
e.g. {"000134": "path/to/dataset/000134.wav"}
"""

import glob
import os

AUDIO_DB_DIR = "./datasets/neural-audio-fp-dataset/music/test-query-db-500-30s/db"
SAVE_PATH = "./datasets/neural-audio-fp-dataset/id2path.txt"

list_file = glob.glob(os.path.join(AUDIO_DB_DIR, "**/*.wav"))

with open(SAVE_PATH, "w") as f:
    for file_path in list_file:
        # get file basename exclude ".wav" part
        file_id = file_path.split("/")[-1][:-4]
        f.write(f"{file_id} {file_path}\n")
print(f"id2path file has been saved to {SAVE_PATH}")
