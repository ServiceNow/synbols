.require-version:
ifndef version
	$(error Please specify a version tag for the docker image (e.g., -e version=v0.0.1.dev1))
endif

docker: .require-version
	docker build \
		-f docker/Dockerfile \
		--target base \
		-t synbols:latest \
		.
	docker tag synbols:latest synbols:$(version)

build_docs:
	sphinx-apidoc -o docs/synbols synbols synbols/run_docker.py
	sphinx-build -aE docs docs/_build/

package: .dev-dependencies
	if [ -d "./build" ]; then rm -r ./build; fi
	if [ -d "./dist" ]; then rm -r ./dist; fi
	python setup.py sdist
	python setup.py bdist_wheel

upload-pypi: .dev-dependencies
	twine upload dist/* --verbose

upload-testpypi: .dev-dependencies
	twine upload --repository testpypi dist/* --verbose

.dev-dependencies:
	pip install wheel twine
