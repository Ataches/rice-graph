
class GraphDataSerializer:
    @staticmethod
    def serialise(d) -> dict:
        return {
            "lineSetFirst": d.lineSetFirst,
            "lineSetSecond": d.lineSetSecond,
            "lineSetThird": d.lineSetThird,
            "lineSetFourth": d.lineSetFourth
        }
