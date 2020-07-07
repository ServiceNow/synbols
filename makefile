package:
	rm -r ./dist ./build
	python setup.py sdist
	python setup.py bdist_wheel