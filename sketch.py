import json
import numpy
import requests
from pprint import pprint

import kass

dims = 28
side = 20

class DrawingBoard(kass.Sketch):
    def setup(self):
        self.width, self.height = 640, 560
        self.no_stroke()
        self.values = numpy.zeros((dims, dims))

    def draw(self):
        for i in range(dims):
            for j in range(dims):
                x, y = i * side, j * side
                d = numpy.sqrt((self.mouseX - x)**2 + (self.mouseY - y)**2)
                if self.mousePressed:
                    val = constrain(self.values[i][j] + numpy.exp(-0.08 * (dims / self.width) * d**2), 0, 1)
                    self.values[i][j] = constrain(val, 0, 1)

                b = 255 * (1 - self.values[i][j])
                self.fill((b, b, b))
                self.rect(x, y, side, side)
        self.button(560, 0, 80, self.reset_values, color=(255, 100, 100))
        self.button(560, 80, 80, self.send_data, color=(0, 100, 100))

    def button(self, x, y, side, function, color = (255,255,255)):
        tmp_color = self._actual_fill_color
        self.fill(color)
        self.rect(x, y, side, side)
        self.fill(tmp_color)
        if self.mousePressed and (abs(x - self.mouseX + 0.5 * side) <= side / 2 and abs(y - self.mouseY + 0.5 * side) <= side / 2):
            function()

    def reset_values(self):
        self.values *= 0

    def send_data(self):
        data = {'values': [ str(x) for x in numpy.reshape(self.values, (784,))]}
        data_json = json.dumps(data)
        payload = {'json_payload': data_json}
        r = requests.get('http://127.0.0.1:5000/evaluate', data=payload)
        pprint()
        return r.json

def constrain(variable, low, high):
    if variable >= high:
        return high
    elif variable <= low:
        return low
    else:
        return variable

if __name__ == "__main__":
    DrawingBoard()