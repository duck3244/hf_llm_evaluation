"""
로깅 유틸리티 모듈
README 예시와 일치하도록 수정된 로거 설정 및 관리
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(name: str, level: str = "INFO",
                 log_file: Optional[str] = None,
                 format_string: Optional[str] = None) -> logging.Logger:
    """로거 설정 (README 예시 구현)"""

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
    """컬러 로그 포매터 (README의 상세 로깅 구현)"""

    # ANSI 컬러 코드
    COLORS = {
        'DEBUG': '\033[36m',    # 청록색
        'INFO': '\033[32m',     # 녹색
        'WARNING': '\033[33m',  # 노란색
        'ERROR': '\033[31m',    # 빨간색
        'CRITICAL': '\033[35m', # 마젠타
        'RESET': '\033[0m'      # 리셋
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']

        # 레벨명에 컬러 적용
        record.levelname = f"{log_color}{record.levelname}{reset_color}"

        return super().format(record)


def setup_colored_logger(name: str, level: str = "INFO") -> logging.Logger:
    """컬러 로거 설정 (README의 --verbose 옵션 구현)"""
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
    """진행률 로깅을 위한 클래스 (README의 배치 수집 구현)"""

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
            f"📊 {self.prefix}: {self.current}/{self.total} "
            f"({percentage:.1f}%) - "
            f"경과: {str(elapsed).split('.')[0]} - "
            f"예상 완료: {eta_str}"
        )

    def finish(self):
        """완료 로깅"""
        elapsed = datetime.now() - self.start_time
        self.logger.info(
            f"✅ {self.prefix} 완료: {self.current}/{self.total} "
            f"(소요시간: {str(elapsed).split('.')[0]})"
        )


def log_function_call(func):
    """함수 호출 로깅 데코레이터 (README의 디버깅 구현)"""

    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"🔧 함수 호출: {func.__name__} with args={args}, kwargs={kwargs}")

        try:
            result = func(*args, **kwargs)
            logger.debug(f"✅ {func.__name__} 성공 완료")
            return result
        except Exception as e:
            logger.error(f"❌ {func.__name__} 실패: {e}")
            raise

    return wrapper


def log_execution_time(func):
    """실행 시간 로깅 데코레이터 (README의 성능 모니터링 구현)"""

    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.now()

        try:
            result = func(*args, **kwargs)
            elapsed = datetime.now() - start_time
            logger.info(f"⏱️  {func.__name__} 실행 완료: {elapsed.total_seconds():.2f}초")
            return result
        except Exception as e:
            elapsed = datetime.now() - start_time
            logger.error(f"❌ {func.__name__} 실패 ({elapsed.total_seconds():.2f}초): {e}")
            raise

    return wrapper


# README 예시에 맞는 프로젝트 로깅 초기화
def init_project_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """프로젝트 전체 로깅 초기화 (README 예시 구현)"""

    # 로그 디렉토리 생성
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # 메인 로그 파일
    timestamp = datetime.now().strftime("%Y%m%d")
    main_log_file = log_path / f"hf_llm_evaluation_{timestamp}.log"

    # 루트 로거 설정 (README 형식)
    root_logger = setup_logger(
        name="hf_llm_evaluation",
        level=log_level,
        log_file=str(main_log_file)
    )

    # 각 모듈별 로거 설정
    modules = [
        "src.api",
        "src.database",
        "src.collectors",
        "src.utils"
    ]

    for module in modules:
        logger = setup_logger(module, level=log_level)
        logger.debug(f"모듈 로거 초기화: {module}")

    root_logger.info("🚀 HuggingFace LLM 평가 프로젝트 로깅 시스템 초기화 완료")
    root_logger.info(f"   • 로그 레벨: {log_level}")
    root_logger.info(f"   • 로그 파일: {main_log_file}")

    return root_logger


# 전역 로거 인스턴스 (README의 사용 패턴과 일치)
_project_logger: Optional[logging.Logger] = None


def get_project_logger() -> logging.Logger:
    """프로젝트 메인 로거 반환 (README 예시 구현)"""
    global _project_logger
    if _project_logger is None:
        _project_logger = init_project_logging()
    return _project_logger


# README의 실시간 모니터링 예시를 위한 추가 함수
def setup_monitoring_logger(name: str = "monitoring") -> logging.Logger:
    """실시간 모니터링용 로거 설정"""
    logger = setup_logger(
        name=name,
        level="INFO",
        log_file=f"logs/monitoring_{datetime.now().strftime('%Y%m%d')}.log"
    )
    return logger


def log_collection_stats(stats: dict, logger: logging.Logger):
    """수집 통계 로깅 (README 형식)"""
    logger.info("📊 수집 통계:")
    logger.info(f"   • 총 모델: {stats.get('total_models', 0):,}개")
    logger.info(f"   • 총 평가 결과: {stats.get('total_evaluations', 0):,}개")
    logger.info(f"   • 성공률: {stats.get('success_rate', 0):.2%}")


def log_api_request(endpoint: str, params: dict, logger: logging.Logger):
    """API 요청 로깅 (README의 API 제한 모니터링)"""
    logger.debug(f"🌐 API 요청: {endpoint}")
    logger.debug(f"   • 파라미터: {params}")


def log_database_operation(operation: str, table: str, count: int, logger: logging.Logger):
    """데이터베이스 작업 로깅"""
    logger.debug(f"💾 DB 작업: {operation} - {table} ({count}개 레코드)")


# README의 문제 해결 섹션을 위한 진단 함수
def diagnose_logging_issues():
    """로깅 관련 문제 진단"""
    issues = []

    # 로그 디렉토리 확인
    log_dir = Path("logs")
    if not log_dir.exists():
        issues.append("로그 디렉토리가 존재하지 않습니다.")
    elif not log_dir.is_dir():
        issues.append("logs가 디렉토리가 아닙니다.")

    # 권한 확인
    try:
        test_file = log_dir / "test.log"
        test_file.touch()
        test_file.unlink()
    except PermissionError:
        issues.append("로그 디렉토리에 쓰기 권한이 없습니다.")

    return issues