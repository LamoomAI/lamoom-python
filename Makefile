PROJECT_FOLDER = 'flow_prompt'

flake8:
	flake8 ${PROJECT_FOLDER}

.PHONY: make-black
make-black:
	black --verbose ${PROJECT_FOLDER}

.PHONY: make-isort
make-isort:
	isort --settings-path pyproject.toml ${PROJECT_FOLDER}

.PHONY: make-mypy
make-mypy:
	mypy --strict ${PROJECT_FOLDER}

isort-check:
	isort --settings-path pyproject.toml --check-only .

autopep8:
	for f in `find flow_prompt -name "*.py"`; do autopep8 --in-place --select=E501 $f; done

test:
	poetry run pytest -vv tests \
		--cov=${PROJECT_FOLDER} \
		--cov-config=.coveragerc \
		--cov-fail-under=76 \
		--cov-report term-missing

.PHONY: format
format: make-black isort-check flake8 make-mypy

clean: clean-build clean-pyc clean-test

clean-build:
		rm -fr build/
		rm -fr dist/
		rm -fr .eggs/
		find . -name '*.egg-info' -exec rm -fr {} +
		find . -name '*.egg' -exec rm -f {} +

clean-pyc:
		find . -name '*.pyc' -exec rm -f {} +
		find . -name '*.pyo' -exec rm -f {} +
		find . -name '*~' -exec rm -f {} +
		find . -name '__pycache__' -exec rm -fr {} +

clean-test:
		rm -f .coverage
		rm -fr htmlcov/
