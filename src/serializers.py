
class GraphDataSerializer:
    @staticmethod
    def serialise(d) -> dict:
        return {
            "lineSet": d.lineSet,
            "min_temp": d.min_temp,
            "max_temp": d.max_temp,
            "rhum": d.rhum,
            "sbright": d.sbright,
            "prec": d.prec,
            "predictedLineSet": d.predicted_line_set,
            "yield_data": d.yield_data,
            "prod": d.prod
        }
