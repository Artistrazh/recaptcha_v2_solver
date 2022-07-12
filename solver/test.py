import requests

info = {
    "text": text,
    "link_to_picture": link_to_picture,
    "squares": squares
}

data = requests.post("http://0.0.0.0:11013/send_big_picture", json=info)
print("data.text", data.text)
