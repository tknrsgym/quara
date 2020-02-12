from typing import List
from quara.objects.elemental_system import ElementalSystem


class State:
    def __init__(self):
        self.elemental_systems: List[ElementalSystem] = []
