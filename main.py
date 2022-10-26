from tkinter.messagebox import NO
from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS
from google.cloud import storage
import classify_service
import json
import io
import time
import datetime

clasify_serv = classify_service.ClasifyService()
bucket = "xirius-landing-ocr-test"
base = "detect"
storage_client = storage.Client()

app = Flask(__name__)
CORS(app)


def download_as_blob(bucket_name, source_blob_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.get_blob(source_blob_name)
    bytes = blob.download_as_bytes()
    print('Object [' + bucket_name + ':' + source_blob_name + '] downloaded')
    return bytes, blob


@app.route('/')
def display_default():
    return 'Welcome to the semantic search app!\n' \
        ''


@app.route('/readiness_check')
def check_readiness():
    return 'App is ready!'


@app.route('/classifyDoc', methods=['POST'])
def clasifyPages():
    resp = {}
    seiz_response = request.json
    resp = clasify_serv.classify(seiz_response)
    return jsonify(resp)


@app.route('/classifyDocWithId', methods=['POST'])
def clasifyPagesWithId():
    resp = {}
    req_json = request.json
    uuid = req_json["uuid"]
    print(uuid)
    blob = None
    time_init = datetime.datetime.now()
    while blob is None and (datetime.datetime.now()-time_init).seconds / 60 < 7:
        blobs = storage_client.list_blobs(bucket, prefix=base+"/"+uuid)
        for b in blobs:
            print(b)
            if ".json" in b.name and not "google" in b.name:
                blob = b
        if(blob is None):
            time.sleep(2)

    if blob is not None:
        bytes = download_as_blob(bucket, blob.name)[0]
        response_json = json.load(io.BytesIO(bytes))
        print(response_json)
        resp = clasify_serv.classify(response_json)
        return jsonify(resp)
    return jsonify("{}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500, debug=True)
