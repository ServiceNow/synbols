SHELL=/bin/bash
userid := $(shell id -u)
versiontag = $(shell date +%Y-%m-%d)

<<<<<<< HEAD
SYNBOLS_RUN = docker run -it -v $(CURDIR)/synbols:/synbols -v $(CURDIR):/local --user `id -u`:`id -g` synbols
=======
SYNBOLS_RUN = docker run --user $(userid) -it -v $(CURDIR)/synbols:/synbols -v $(CURDIR):/local synbols
>>>>>>> origin/master

all: docker explore-fonts view-dataset

docker:
	docker build -t synbols:latest .
	docker tag synbols:latest synbols:$(versiontag)
explore-fonts:
	$(SYNBOLS_RUN) sh -c "cd /local; python ../synbols/explore_fonts.py"
view-dataset:
	$(SYNBOLS_RUN) sh -c "cd /local; python ../synbols/view_dataset.py"
	open dataset.png
dataset:
	$(SYNBOLS_RUN) sh -c "cd /local; python ../synbols/generate_dataset.py"
