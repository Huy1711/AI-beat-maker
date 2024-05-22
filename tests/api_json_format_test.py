import json


def gen_song_json_test():
    with open("samples/gen_song_format.json", "r") as f:
        data = json.load(f)
        error_message = data["metadata"]["error_message"]
        if error_message is not None:
            raise Exception(error_message)
        for song in data["clips"]:
            song_id = song["id"]
            print(song_id)


def get_song_json_test():
    with open("samples/get_song_format.json", "r") as f:
        data = json.load(f)
        print(data[0]["audio_url"])
        # error_message = data["metadata"]["error_message"]
        # if error_message is not None:
        #     raise Exception(error_message)
        # for song in data["clips"]:
        #     song_id = song["id"]
        #     print(song_id)


if __name__ == "__main__":
    get_song_json_test()
