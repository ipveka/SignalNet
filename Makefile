install:
	pip install -r requirements.txt

test:
	pytest --cov=src/signalnet tests/

lint:
	mypy src/signalnet

docs:
	cd docs && make html

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache docs/_build
