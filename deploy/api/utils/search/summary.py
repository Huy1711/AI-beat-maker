import typing as tp
from itertools import groupby

import numpy as np

id2path_dict = {}
with open("./datasets/neural-audio-fp-dataset/id2path.txt", "r") as f:
    for line in f:
        file_id, path = line.split()
        id2path_dict[file_id] = path


def get_unique_candidates(search_results) -> tp.List[tp.Tuple]:
    """
    get candidate with unique result of (pred_song_id, pred_song_start)
        with pred_song_start = pred_song_offset - current_index

    Returns
        A list of candidates with the format
         (index, score, pred_song_start, pred_song_id)
    """
    candidates = []
    for current_offset, segment in enumerate(search_results):
        pred_song_id = segment[0]["entity"]["file_id"]
        pred_song_start = segment[0]["entity"]["offset"] - current_offset
        # This is similarity score not distance as inner product metric is used
        pred_score = segment[0]["distance"]
        candidates.append((current_offset, pred_score, pred_song_start, pred_song_id))
    return candidates


def filter_and_format_result(candidates, mean_score_thresh=0.5, seconds_thresh=3.0):
    """
    Grouping segments by unique (pred_song_id, pred_song_start)
    Elements in a group should be consecutive by index and consistent in prediction
    e.g. (index, score, pred_song_start, pred_song_id)
         [(12, 0.6421586275100708, 0, '001028')
          (13, 0.5211837291717529, 0, '001028')
          (14, 0.6640710830688477, 0, '001028')]
    """

    segments_thresh = int(seconds_thresh * 2 - 1)  # e.g. 3s audio ~ 5 embeddings
    detected_chunks = []
    for _, group in groupby(candidates, key=lambda x: (x[-1], x[-2])):
        group = list(group)
        mean_score = np.mean([segment[1] for segment in group])
        if len(group) < segments_thresh or mean_score < mean_score_thresh:
            continue

        start_frame, end_frame = group[0][0], group[-1][0]
        chunk_report = {
            "start": start_frame / 2,  # start second in the query audio
            "duration": (end_frame - start_frame + 1) / 2 + 0.5,
            "file_id": group[0][3],
            "score": mean_score,
            "enrolled_start": (group[0][2] + start_frame) / 2,
        }
        detected_chunks.append(chunk_report)
    return detected_chunks


def summary_result(search_results, mean_score_thresh=0.5, seconds_thresh=3.0):
    candidates = get_unique_candidates(search_results)
    final_result = filter_and_format_result(
        candidates, mean_score_thresh, seconds_thresh
    )
    return final_result


def get_song_by_ids(file_id):
    return id2path_dict[file_id]
