import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


class Config:
    """프로젝트 설정 클래스"""

    # API 설정
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    HUGGINGFACE_API_BASE = "https://huggingface.co/api"

    # 데이터베이스 설정
    DATABASE_PATH = "data/llm_evaluations.db"

    # 디렉토리 설정
    REPORTS_DIR = "reports"
    EXPORTS_DIR = "exports"

    # 수집할 태스크 목록 (README의 지원 태스크와 일치)
    TASKS_TO_COLLECT = [
        "text-generation",
        "text-classification",
        "question-answering",
        "summarization",
        "translation"
    ]

    # 태스크별 설정 (README 예시와 일치)
    MODELS_PER_TASK = int(os.getenv("MODELS_PER_TASK", 30))
    MAX_EVALUATIONS_PER_MODEL = 10

    # API 요청 설정 (README 예시와 일치)
    API_DELAY = float(os.getenv("API_DELAY", 0.1))  # 초
    REQUEST_TIMEOUT = 30  # 초
    MAX_RETRIES = 3

    # 태스크 카테고리 정의 (README 테이블과 정확히 일치)
    TASK_CATEGORIES = {
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

    # 로깅 설정 (README 예시와 일치)
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def validate(cls):
        """설정 검증"""
        required_dirs = [cls.REPORTS_DIR, cls.EXPORTS_DIR, "data"]
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def get_task_info(cls, task_name: str) -> Dict[str, Any]:
        """특정 태스크의 정보 반환"""
        return cls.TASK_CATEGORIES.get(task_name, {})