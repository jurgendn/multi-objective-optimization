class ResultReporter:
    def __init__(self) -> None:
        self.xmin = []
        self.ymin = []

    def add_efficient_point(self, x_min, y_min):
        self.xmin.append(x_min)
        self.ymin.append(y_min)

    def __repr__(self) -> str:
        s = []
        for k in self.__dict__:
            v = self.__getattribute__(k)
            s.append(f"{k}:\t\t{v}")
        return "\n".join(s)

    def pareto_front(self):
        pass
