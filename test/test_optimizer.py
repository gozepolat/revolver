import unittest
from stacked.meta.heuristics import population
from stacked.utils import transformer, common
from PIL import Image
import glob


class TestTrainer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTrainer, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        pass
