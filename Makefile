.PHONY: clean test docs

export SPHINX_MOCK_REQUIREMENTS=1
# ref: pytorch-lightning

clean:
	# clean all temp runs
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf ./docs/build
	rm -rf ./docs/source/generated
	rm -rf ./docs/source/*/generated
	rm -rf ./docs/source/api

test: clean
	# Review the CONTRIBUTING docmentation for other ways to test.
	pip install -r requirements/dev.txt

	# run tests with coverage
	#TODO
	python -m coverage run --source pytorch_lightning -m pytest pytorch_lightning tests pl_examples -v
	python -m coverage report

docs: clean
	pip3 install --quiet -r requirements/docs.txt
	python3 -m sphinx -b html -W docs docs/_build