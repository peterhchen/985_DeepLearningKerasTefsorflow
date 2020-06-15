from flask import request
from flask import jsonify
from flask import Flask

app = Flask (__name__)

# parse the route and method from client.
@app.route ('/hello', methods = ['POST'])
def hello ():
    message = request.get_json (force=True)
    # Mesage has to contains the key 'name' and value associate key.
    # value = message[key]
    name = message['name']
    response = {
        # send back with JSON format with 
        # key: value
        'greeting': 'Hello, ' + name + '!'
    }
    # return with json format.
    return jsonify (response)
    