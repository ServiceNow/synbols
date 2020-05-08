# Synbols benchmarks
The entry point is in `trainval.py`. It takes care of checkpointing and calling the training and validation loops

## Requirements
```
$ pip install --upgrade git+https://github.com/ElementAI/haven
```

## Add benchmark
Individual benchmarks should be placed into `models`.
# TODO

## Few Shot


1. Figure out bug when MAML is in eval()
2. Add Meta-SGD
3. Add new args for MAML
4. Code hparam search + launch
