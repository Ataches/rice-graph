from src.models import GraphData


def start(msg):
    graphData = {"lineSetFirst": {
            'ENE': 5.05,
            'FEB': 4.8,
            'MAR': 4.7
        },
        "lineSetSecond": {
            'ABR': 4,
            'MAY': 4,
            'JUN': 4.7
        },
        "lineSetThird": {
            'JUL': 5.8,
            'AGO': 4.2,
            'SEPT': 4.1
        },
        "lineSetFourth": {
            'OCT': 5.05,
            'NOV': 4.8,
            'DIC': 2
        }}
    return GraphData(graphData)
