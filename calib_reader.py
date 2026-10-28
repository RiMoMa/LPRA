"""CalibrationDataReader: the only interface ORT's calibrator has to our data.

ORT calls get_next() until it returns None, feeding each dict to an augmented copy of the model whose
every intermediate tensor is a graph output. Whatever we yield here IS the distribution the
activation ranges are fit to, hence common.preprocess (identical letterbox + 114 padding + /255).
"""
import cv2
from onnxruntime.quantization import CalibrationDataReader
from common import preprocess

class CocoCalibReader(CalibrationDataReader):
    def __init__(self, image_paths, input_name="images"):
        self.paths, self.input_name, self.i = list(image_paths), input_name, 0

    def get_next(self):
        if self.i >= len(self.paths):
            return None
        x, _ = preprocess(cv2.imread(self.paths[self.i]))   # batch 1: MinMax reduces across calls anyway
        self.i += 1
        return {self.input_name: x}

    def rewind(self):   # lets one reader instance be reused for several variants
        self.i = 0
