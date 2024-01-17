from collections.abc import Callable

import numpy as np

class RotaSolve:

    def __init__(self, callback: Callable | None = None) -> None:
        if callback is not None:
            self._callback = callback
        self.max_iterations = 5


    def minimize(self, f: Callable[[list[float]], float], x: list[float], jac=None):
        for iteration in range(self.max_iterations):
            phi = 0.0
            for i in range(len(x)):
                x0 = x.copy()
                x0[i] = phi
                f0 = f(x0)
                x0[i] = phi + np.pi/2
                fp = f(x0)
                x0[i] = phi - np.pi/2
                fm = f(x0)
                x[i] = phi - np.pi/2 - np.arctan2(2*f0 - fp - fm, fp - fm)
            print(iteration, f(x), x)



