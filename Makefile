venv:
	virtualenv --python=python3.7 venv && venv/bin/python -m pip install -e .

run:
	venv/bin/python vulcan/main.py