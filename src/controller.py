from flask import Flask, jsonify
from flask import request
from models import GraphData
import json

from src.graph_processing import GraphProcessing
from src.serializers import GraphDataSerializer

graph_processing = GraphProcessing()
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


@app.route("/", methods=['POST'])
def post_method():
    response: GraphData = graph_processing.graph_data_by_variety(json.loads(request.data))
    return jsonify(GraphDataSerializer.serialise(response))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)