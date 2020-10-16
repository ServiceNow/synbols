# Version tag as command argument
.require-version:
ifndef version
	$(error Please specify a version tag for the docker image (e.g., -e version=v0.0.1.dev1))
endif

# Builds the docker image
docker: .require-version
	cp -r ./synbols docker
	cp -r ./developer_tools docker
	docker build \
		-f docker/Dockerfile \
		--target base \
		-t synbols:latest \
		.
	docker tag synbols:latest synbols:$(version)
	rm -r ./docker/synbols ./docker/developer_tools

# Builds the documentation
build_docs:
	cd docs && sphinx-build -aE . _build/

# Builds the font blacklist and outputs files to the current directory
font-blacklist: .require-version
	$(eval TMP_WD=$(TMPDIR)/synbols_font_blacklist)
	@mkdir -p $(TMP_WD)
	@echo "FROM aldro61/synbols:$(version)\nRUN pip install --upgrade pip && pip install pandas torch torchvision" > $(TMP_WD)/Dockerfile
	@cd $(TMP_WD) && docker build -t "synbols:$(version)-fontblacklist" . && cd -
	@SYNBOLS_DEV_IMAGE="synbols:$(version)-fontblacklist" \
	synbols developer_tools/blacklist_fonts.py --mount-path=/tmp

# Package code for release to PyPI
package: .dev-dependencies
	if [ -d "./build" ]; then rm -r ./build; fi
	if [ -d "./dist" ]; then rm -r ./dist; fi
	python setup.py sdist
	python setup.py bdist_wheel

# Upload package to PyPI (official repo)
upload-pypi: .dev-dependencies
	twine upload dist/* --verbose

# Upload package to PyPI (test repo)
upload-testpypi: .dev-dependencies
	twine upload --repository testpypi dist/* --verbose

# Dependencies for uploading to PyPI
.dev-dependencies:
	pip install wheel twine

# Runs all tests
run_tests:
	pytest tests

# Runs flake8
run_flake8:
	flake8 synbols tests
