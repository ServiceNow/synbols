from . import classification
from . import fewshot
from . import active_learning

EXP_GROUPS = {}
EXP_GROUPS.update(classification.EXP_GROUPS)
EXP_GROUPS.update(fewshot.EXP_GROUPS)
EXP_GROUPS.update(active_learning.EXP_GROUPS)