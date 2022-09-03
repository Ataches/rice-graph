from flask import Flask, jsonify
from flask import request
from models import GraphData
import graph_processing
import json

from src.serializers import GraphDataSerializer

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


@app.route("/", methods=['POST'])
def post_method():
    response: GraphData = graph_processing.start(json.loads(request.data))
    return jsonify(GraphDataSerializer.serialise(response))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)