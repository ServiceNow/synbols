#!/bin/bash

tmux new-session -s "main-datasets" -d 'synbols-datasets --n_samples=100000'
tmux split-window -v 'synbols-datasets --n_samples=100000 --dataset=camouflage'
tmux split-window -v 'synbols-datasets --n_samples=100000 --dataset=korean-1k'
tmux split-window -v 'synbols-datasets --n_samples=100000 --dataset=less-variations'

tmux new-session -s "active-learning-datasets" -d 'synbols-datasets --n_samples=100000 --dataset=pixel-noise'
tmux split-window -v 'synbols-datasets --n_samples=100000 --dataset=some-large-occlusion'
tmux split-window -v 'synbols-datasets --n_samples=100000 --dataset=missing-symbol'
tmux split-window -v 'synbols-datasets --n_samples=100000 --dataset=large-translation'

tmux new-session -s "big-datasets" -d 'synbols-datasets --n_samples=1000000 --dataset=default'
tmux split-window -v 'synbols-datasets --n_samples=1000000 --dataset=balanced-font-chars'

tmux new-session -s "gen-datasets" -d 'synbols-datasets --n_samples=100000 --dataset=non-camou-bw'
tmux split-window -v 'synbols-datasets --n_samples=100000 --dataset=non-camou-shade'

tmux new-session -s "counting-datasets" -d 'synbols-datasets --n_samples=100000 --dataset=counting'
tmux split-window -v 'synbols-datasets --n_samples=100000 --dataset=counting-fix-scale'
tmux split-window -v 'synbols-datasets --n_samples=100000 --dataset=counting-crowded'


tmux -2 attach-session -d