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
