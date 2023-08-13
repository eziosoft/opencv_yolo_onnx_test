import time


class FPSCounter:
    def __init__(self):
        self.timeMark = time.time()
        self.dtFilter = 0
        self.fps = 0

    def update(self):
        dt = time.time() - self.timeMark
        self.timeMark = time.time()
        self.dtFilter = 0.9 * self.dtFilter + 0.1 * dt
        self.fps = 1 / self.dtFilter

    def getFPS(self):
        return self.fps
