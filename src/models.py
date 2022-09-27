
class GraphData:

    def __init__(self, graph_data):
        self.lineSet: dict = graph_data["lineSet"]
        self.min_temp: float = graph_data["min_temp"]
        self.max_temp: float = graph_data["max_temp"]
        self.rhum: float = graph_data["rhum"]
        self.sbright: float = graph_data["sbright"]
        self.prec: float = graph_data["prec"]
        self.prod: float = graph_data["prod"]

