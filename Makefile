
all: test integ

test:
	pytest ./tests/unit
	pytest --doctest-modules ./pydcop
	pytest ./tests/dcop_cli
	pytest ./tests/api

test_cli:
	pytest ./tests/dcop_cli

test_unit:
	pytest ./tests/unit
	pytest --doctest-modules ./pydcop


test_api:
	pytest ./tests/api

mypy:
	 mypy --ignore-missing-imports  pydcop


coverage: 
	coverage run --source=. -m unittest discover ./tests/unit
	coverage report

doc: 
	python -m sphinx ./docs ./docs/_build/

