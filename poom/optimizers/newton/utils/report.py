class ResultReporter:
    def __init__(self, x_min, y_min, n_iter) -> None:
        self.argmin = x_min
        self.min = y_min
        self.n_iter = n_iter

    def __repr__(self) -> str:
        s = []
        for k in self.__dict__:
            v = self.__getattribute__(k)
            s.append(f"{k}:\t\t{v}")
        return "\n".join(s)