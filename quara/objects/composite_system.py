from typing import List
from quara.objects.elemental_system import ElementalSystem


class CompositeSystem:
    """合成系を記述するためのクラス"""

    def __init__(self):
        self.elemental_systems: List[ElementalSystem] = []
