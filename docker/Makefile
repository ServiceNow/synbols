docker: .require-version
	docker build -t synbols:latest .
	docker tag synbols:latest synbols:$(version)

.require-version:
ifndef version
	$(error Please specify a version tag for the docker image (e.g., version=v0.0.1.dev1))
endif