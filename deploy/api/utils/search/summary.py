from collections import Counter

import soundfile as sf

id2path_dict = {}
with open("./datasets/neural-audio-fp-dataset/id2path.txt", "r") as f:
    for line in f:
        file_id, path = line.split()
        id2path_dict[file_id] = path


def summary_result(search_results, score_thresh=0.5, seconds_thresh=5.0):
    # number_of_segments_allow = seconds_thresh * 2 - 1

    guessed_file_ids = Counter(
        [segment[0]["entity"]["file_id"] for segment in search_results]
    )
    major_voting_result = guessed_file_ids.most_common(1)[0][0]
    return major_voting_result


def get_song_by_ids(file_id):
    return id2path_dict[file_id]
