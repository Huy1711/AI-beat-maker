import requests

url = "http://0.0.0.0:8080/search"
audio_files = "samples/001001.wav"
resp = requests.post(url=url, files={"file": open(audio_files, "rb")})
