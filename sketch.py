import numpy
import requests

import kass

dims = 28
side = 20
values = numpy.zeros((dims, dims))


class DrawingBoard(kass.Sketch):
    def setup(self):
        self.width, self.height = 640, 560
        self.no_stroke()

    def draw(self):
        for i in range(dims):
            for j in range(dims):
                x, y = i * side, j * side
                d = numpy.sqrt((self.mouseX - x)**2 + (self.mouseY - y)**2)
                if self.mousePressed:
                    val = constrain(values[i][j] + numpy.exp(-0.08 * (dims / self.width) * d**2), 0, 1)
                    values[i][j] = constrain(val, 0, 1)

                b = 255 * (1 - values[i][j])
                self.fill((b, b, b))
                self.rect(x, y, side, side)

def constrain(variable, low, high):
    if variable >= high:
        return high
    elif variable <= low:
        return low
    else:
        return variable

if __name__ == "__main__":
    DrawingBoard()