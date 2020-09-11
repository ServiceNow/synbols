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

run_tests:
	pytest tests

run_flake8:
	flake8 synbols tests
