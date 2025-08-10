"""
로깅 유틸리티 모듈
프로젝트 전체에서 사용할 로거 설정 및 관리
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(name: str, level: str = "INFO",
                 log_file: Optional[str] = None,
                 format_string: Optional[str] = None) -> logging.Logger:
    """로거 설정"""

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 로거 생성
    logger = logging.getLogger(name)

    # 기존 핸들러 제거 (중복 방지)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 로그 레벨 설정
    logger.setLevel(getattr(logging, level.upper()))

    # 포매터 생성
    formatter = logging.Formatter(format_string)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러 (옵션)
    if log_file:
        # 로그 디렉토리 생성
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """로거 인스턴스 반환"""
    return logging.getLogger(name)


class ColoredFormatter(logging.Formatter):
    """컬러 로그 포매터"""

    # ANSI 컬러 코드
    COLORS = {
        'DEBUG': '\033[36m',  # 청록색
        'INFO': '\033[32m',  # 녹색
        'WARNING': '\033[33m',  # 노란색
        'ERROR': '\033[31m',  # 빨간색
        'CRITICAL': '\033[35m',  # 마젠타
        'RESET': '\033[0m'  # 리셋
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']

        # 레벨명에 컬러 적용
        record.levelname = f"{log_color}{record.levelname}{reset_color}"

        return super().format(record)


def setup_colored_logger(name: str, level: str = "INFO") -> logging.Logger:
    """컬러 로거 설정"""
    logger = logging.getLogger(name)

    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(getattr(logging, level.upper()))

    # 컬러 포매터
    formatter = ColoredFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


class ProgressLogger:
    """진행률 로깅을 위한 클래스"""

    def __init__(self, total: int, logger: logging.Logger,
                 prefix: str = "Progress", interval: int = 10):
        self.total = total
        self.current = 0
        self.logger = logger
        self.prefix = prefix
        self.interval = interval
        self.start_time = datetime.now()

    def update(self, increment: int = 1):
        """진행률 업데이트"""
        self.current += increment

        if self.current % self.interval == 0 or self.current == self.total:
            self._log_progress()

    def _log_progress(self):
        """진행률 로깅"""
        percentage = (self.current / self.total) * 100
        elapsed = datetime.now() - self.start_time

        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = str(eta).split('.')[0]  # 소수점 제거
        else:
            eta_str = "Unknown"

        self.logger.info(
            f"{self.prefix}: {self.current}/{self.total} "
            f"({percentage:.1f}%) - "
            f"Elapsed: {str(elapsed).split('.')[0]} - "
            f"ETA: {eta_str}"
        )

    def finish(self):
        """완료 로깅"""
        elapsed = datetime.now() - self.start_time
        self.logger.info(
            f"{self.prefix} completed: {self.current}/{self.total} "
            f"in {str(elapsed).split('.')[0]}"
        )


def log_function_call(func):
    """함수 호출 로깅 데코레이터"""

    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")

        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise

    return wrapper


def log_execution_time(func):
    """실행 시간 로깅 데코레이터"""

    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.now()

        try:
            result = func(*args, **kwargs)
            elapsed = datetime.now() - start_time
            logger.info(f"{func.__name__} executed in {elapsed.total_seconds():.2f} seconds")
            return result
        except Exception as e:
            elapsed = datetime.now() - start_time
            logger.error(f"{func.__name__} failed after {elapsed.total_seconds():.2f} seconds: {e}")
            raise

    return wrapper


# 기본 로거 설정
def init_project_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """프로젝트 전체 로깅 초기화"""

    # 로그 디렉토리 생성
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # 메인 로그 파일
    timestamp = datetime.now().strftime("%Y%m%d")
    main_log_file = log_path / f"hf_llm_evaluation_{timestamp}.log"

    # 루트 로거 설정
    root_logger = setup_logger(
        name="hf_llm_evaluation",
        level=log_level,
        log_file=str(main_log_file)
    )

    # 각 모듈별 로거 설정
    modules = [
        "hf_llm_evaluation.api",
        "hf_llm_evaluation.database",
        "hf_llm_evaluation.collectors",
        "hf_llm_evaluation.utils"
    ]

    for module in modules:
        logger = setup_logger(module, level=log_level)
        logger.info(f"Logger initialized for {module}")

    root_logger.info("Project logging initialized")
    return root_logger


# 전역 로거 인스턴스 (지연 초기화)
_project_logger: Optional[logging.Logger] = None


def get_project_logger() -> logging.Logger:
    """프로젝트 메인 로거 반환"""
    global _project_logger
    if _project_logger is None:
        _project_logger = init_project_logging()
    return _project_logger