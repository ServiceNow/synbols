SHELL=/bin/bash
userid := $(shell id -u)
versiontag = $(shell date +%Y-%m-%d)

SYNBOLS_RUN = docker run --user $(userid) -it -v $(CURDIR)/generator:/generator -v $(CURDIR):/local synbols

docker:
	docker build -t synbols:latest .
	docker tag synbols:latest synbols:$(versiontag)

font-cache:
	rm alphabet_fonts.cache; $(SYNBOLS_RUN) sh -c "cd /local; python -c 'from synbols.fonts import ALPHABET_MAP'"

view-dataset:
	$(SYNBOLS_RUN) sh -c "cd /local; python ../generator/view_dataset.py"
	open dataset.png

dataset:
	$(SYNBOLS_RUN) sh -c "cd /local; python ../generator/generate_dataset.py"

test:
	$(SYNBOLS_RUN) sh -c "cd /local; python ../generator/view_generator.py"

view_font_clusters:
	$(SYNBOLS_RUN) sh -c "cd /local; python ../generator/view_font_clustering.py"

datasets:
	generator/generate_dataset.py --n_samples=100000 &
	generator/generate_dataset.py --n_samples=1000000 &
	generator/generate_dataset.py --dataset=camouflage --n_samples=100000 &
	generator/generate_dataset.py --dataset=tiny --n_samples=10000 &
	generator/generate_dataset.py --dataset=segmentation --n_samples=100000 &
	generator/generate_dataset.py --dataset=missing-symbol --n_samples=100000 &
	generator/generate_dataset.py --dataset=all_languages --n_samples=1000000 &
	wait

splits:
	$(SYNBOLS_RUN) sh -c "cd /local; python ../generator/generate_splits.py"


font_check:
	$(SYNBOLS_RUN) sh -c "cd /local; python generator/run_font_checks.py"