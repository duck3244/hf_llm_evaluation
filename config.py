import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


# 설정을 함수 기반으로 변경하여 순환 임포트 방지
def get_huggingface_token() -> str:
    """HuggingFace API 토큰 반환"""
    return os.getenv("HUGGINGFACE_TOKEN", "")


def get_huggingface_api_base() -> str:
    """HuggingFace API 기본 URL 반환"""
    return "https://huggingface.co/api"


def get_database_path() -> str:
    """데이터베이스 파일 경로 반환"""
    return "data/llm_evaluations.db"


def get_reports_dir() -> str:
    """리포트 디렉토리 경로 반환"""
    return "reports"


def get_exports_dir() -> str:
    """내보내기 디렉토리 경로 반환"""
    return "exports"


def get_tasks_to_collect() -> List[str]:
    """수집할 태스크 목록 반환"""
    return [
        "text-generation",
        "text-classification",
        "question-answering",
        "summarization",
        "translation"
    ]


def get_models_per_task() -> int:
    """태스크별 수집할 모델 수 반환"""
    return int(os.getenv("MODELS_PER_TASK", 30))


def get_max_evaluations_per_model() -> int:
    """모델당 최대 평가 결과 수 반환"""
    return 10


def get_api_delay() -> float:
    """API 요청 간격 반환"""
    return float(os.getenv("API_DELAY", 0.1))


def get_request_timeout() -> int:
    """API 요청 타임아웃 반환"""
    return 30


def get_max_retries() -> int:
    """최대 재시도 횟수 반환"""
    return 3


def get_task_categories() -> Dict[str, Dict[str, Any]]:
    """태스크 카테고리 정보 반환"""
    return {
        "text-generation": {
            "description": "텍스트 생성",
            "common_datasets": ["hellaswag", "arc", "mmlu"],
            "common_metrics": ["accuracy", "perplexity"]
        },
        "text-classification": {
            "description": "텍스트 분류",
            "common_datasets": ["glue", "imdb", "sst2"],
            "common_metrics": ["accuracy", "f1"]
        },
        "question-answering": {
            "description": "질문 답변",
            "common_datasets": ["squad", "natural_questions"],
            "common_metrics": ["exact_match", "f1"]
        },
        "summarization": {
            "description": "요약",
            "common_datasets": ["cnn_dailymail", "xsum"],
            "common_metrics": ["rouge-1", "rouge-2"]
        },
        "translation": {
            "description": "번역",
            "common_datasets": ["wmt14", "opus"],
            "common_metrics": ["bleu", "meteor"]
        }
    }


def get_log_level() -> str:
    """로그 레벨 반환"""
    return os.getenv("LOG_LEVEL", "INFO")


def get_log_format() -> str:
    """로그 포맷 반환"""
    return "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_task_info(task_name: str) -> Dict[str, Any]:
    """특정 태스크의 정보 반환"""
    return get_task_categories().get(task_name, {})


def validate_config():
    """설정 검증 및 디렉토리 생성"""
    required_dirs = [get_reports_dir(), get_exports_dir(), "data"]
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)


# 하위 호환성을 위한 클래스 래퍼 (기존 코드와의 호환성 유지)
class Config:
    """하위 호환성을 위한 설정 클래스"""

    @property
    def HUGGINGFACE_TOKEN(self):
        return get_huggingface_token()

    @property
    def HUGGINGFACE_API_BASE(self):
        return get_huggingface_api_base()

    @property
    def DATABASE_PATH(self):
        return get_database_path()

    @property
    def REPORTS_DIR(self):
        return get_reports_dir()

    @property
    def EXPORTS_DIR(self):
        return get_exports_dir()

    @property
    def TASKS_TO_COLLECT(self):
        return get_tasks_to_collect()

    @property
    def MODELS_PER_TASK(self):
        return get_models_per_task()

    @property
    def MAX_EVALUATIONS_PER_MODEL(self):
        return get_max_evaluations_per_model()

    @property
    def API_DELAY(self):
        return get_api_delay()

    @property
    def REQUEST_TIMEOUT(self):
        return get_request_timeout()

    @property
    def MAX_RETRIES(self):
        return get_max_retries()

    @property
    def TASK_CATEGORIES(self):
        return get_task_categories()

    @property
    def LOG_LEVEL(self):
        return get_log_level()

    @property
    def LOG_FORMAT(self):
        return get_log_format()

    def get_task_info(self, task_name: str) -> Dict[str, Any]:
        """특정 태스크의 정보 반환"""
        return get_task_info(task_name)

    def validate(self):
        """설정 검증"""
        validate_config()