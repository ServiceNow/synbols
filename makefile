SHELL=/bin/bash
userid := $(shell id -u)
versiontag = $(shell date +%Y-%m-%d)

SYNBOLS_RUN = docker run --user $(userid) -it -v $(CURDIR)/generator:/generator -v $(CURDIR):/local synbols

all: docker explore-fonts view-dataset

docker:
	docker build -t synbols:latest .
	docker tag synbols:latest synbols:$(versiontag)
font-cache:
	$(SYNBOLS_RUN) sh -c "cd /local; python -c 'from synbols.fonts import ALPHABET_MAP'"


explore-fonts:
	$(SYNBOLS_RUN) sh -c "cd /local; python ../synbols/explore_fonts.py"
view-dataset:
	$(SYNBOLS_RUN) sh -c "cd /local; python ../synbols/view_dataset.py"
	open dataset.png
dataset:
	$(SYNBOLS_RUN) sh -c "cd /local; python ../synbols/generate_dataset.py"
test:
	$(SYNBOLS_RUN) sh -c "cd /local; python ../generator/view_generator.py"
