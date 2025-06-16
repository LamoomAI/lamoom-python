PROJECT_FOLDER = 'lamoom'

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
	for f in `find lamoom -name "*.py"`; do autopep8 --in-place --select=E501 $f; done

lint:
	poetry run isort --settings-path pyproject.toml --check-only .

test:
	poetry run pytest --cache-clear -vv tests

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
		rm -rf .pytest_cache


publish-test-prerelease:
	poetry version prerelease
	poetry build
	twine upload --repository testpypi dist/*


publish-release:
	poetry config pypi-token.pypi "$(PYPI_API_KEY)"
	poetry version patch
	poetry build
	poetry publish
