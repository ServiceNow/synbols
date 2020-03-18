from . import classification
from . import fewshot

EXP_GROUPS = {}
EXP_GROUPS.update(classification.EXP_GROUPS)
EXP_GROUPS.update(fewshot.EXP_GROUPS)