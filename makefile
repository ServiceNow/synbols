versiontag = $(shell date +%Y-%m-%d)

all: docker explore-fonts view-dataset

docker:
	docker build -t synbols:latest .
	docker tag synbols:latest synbols:$(versiontag)
explore-fonts:
	docker run -it -v $(CURDIR)/synbols:/synbols -v $(CURDIR):/local synbols sh -c "cd /local; python ../synbols/explore_fonts.py"
view-dataset:
	docker run -it -v $(CURDIR)/synbols:/synbols -v $(CURDIR):/local synbols sh -c "cd /local; python ../synbols/view_dataset.py"
	open dataset.png
