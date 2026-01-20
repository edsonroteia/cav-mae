# -*- coding: utf-8 -*-
# @Time    : 6/19/21 4:31 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : __init__.py

from .cav_mae import CAVMAE, CAVMAEFT
from .audio_mdl import CAVMAEFTAudio

# CAV-JEPA imports are optional (only available on cav-mae-jepa branch)
try:
    from .cav_jepa import CAVJEPA, CAVJEPAFT
except ImportError:
    CAVJEPA = None
    CAVJEPAFT = None