.PHONY: help install install-dev test test-cov lint format clean run

help:
	@echo "사용 가능한 명령어:"
	@echo "  install     - 프로젝트 의존성 설치"
	@echo "  install-dev - 개발 의존성 포함 설치"
	@echo "  test        - 테스트 실행"
	@echo "  test-cov    - 커버리지와 함께 테스트 실행"
	@echo "  lint        - 코드 린팅"
	@echo "  format      - 코드 포맷팅"
	@echo "  clean       - 임시 파일 정리"
	@echo "  run         - 프로젝트 실행"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e .[dev]

test:
	pytest tests/

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ main.py config.py
	mypy src/ main.py config.py

format:
	black src/ main.py config.py
	isort src/ main.py config.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/ dist/ *.egg-info/
	rm -rf .coverage htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

run:
	python main.py --stats

collect-all:
	python main.py --collect-all

collect-text-gen:
	python main.py --task text-generation

update:
	python main.py --update

reports:
	python main.py --reports

export:
	python main.py --export