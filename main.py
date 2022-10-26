from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS
import classify_service
import json

clasify_serv = classify_service.ClasifyService()

app = Flask(__name__)
CORS(app)

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

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5500, debug=True)
