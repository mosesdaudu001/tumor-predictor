import requests

resp = requests.post("https://us-central1-idyllic-kiln-407416.cloudfunctions.net/function-1", files={'file': open('011_big_gallery.jpeg', 'rb')})
# resp = requests.post("http://127.0.0.1:8000/predict", files={'file': open('011_big_gallery.jpeg', 'rb')})

print(resp)
# print(resp.json())
