#!/bin/bash

tmux new-session -s "generate-datasets" -d 'synbols-datasets --n_samples=100000'
tmux split-window -v 'synbols-datasets --n_samples=100000 --dataset=camouflage'
tmux split-window -v 'synbols-datasets --n_samples=100000 --dataset=korean-1k'
tmux split-window -v 'synbols-datasets --n_samples=100000 --dataset=less-variations'
tmux split-window -v 'synbols-datasets --n_samples=100000 --dataset=pixel-noise'
tmux split-window -v 'synbols-datasets --n_samples=100000 --dataset=some-large-occlusion'
tmux split-window -v 'synbols-datasets --n_samples=100000 --dataset=missing-symbol'
tmux split-window -v 'synbols-datasets --n_samples=100000 --dataset=large-translation'


tmux -2 attach-session -d