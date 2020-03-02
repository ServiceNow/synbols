versiontag = $(shell date +%Y-%m-%d)

SYNBOLS_RUN = docker run -it -v $(CURDIR)/synbols:/synbols -v $(CURDIR):/local synbols

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
