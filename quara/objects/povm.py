from typing import List
from quara.engine.matlabengine import MatlabEngine
from quara.objects.elemental_system import ElementalSystem


class Povm:
    def __init__(self):
        self.elemental_systems: List[ElementalSystem] = []

