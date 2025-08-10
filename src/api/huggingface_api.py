"""
HuggingFace API 클라이언트 모듈
HuggingFace Hub API와의 통신을 담당
"""

import requests
import time
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urljoin
import logging

from ..models.data_models import ModelInfo, EvaluationResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


class HuggingFaceAPIError(Exception):
    """HuggingFace API 관련 예외"""
    pass


class RateLimitError(HuggingFaceAPIError):
    """API 요청 제한 예외"""
    pass


class HuggingFaceAPI:
    """HuggingFace API 클라이언트"""

    def __init__(self, token: Optional[str] = None, base_url: str = "https://huggingface.co/api"):
        self.base_url = base_url
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "HF-LLM-Evaluation-Collector/1.0"
        }

        if token:
            self.headers["Authorization"] = f"Bearer {token}"
            logger.info("HuggingFace API 토큰이 설정되었습니다.")
        else:
            logger.warning("HuggingFace API 토큰이 설정되지 않았습니다. 일부 기능이 제한될 수 있습니다.")

        self.session.headers.update(self.headers)

    def _make_request(self, endpoint: str, params: Optional[Dict] = None,
                      max_retries: int = 3, delay: float = 0.1) -> Dict[str, Any]:
        """API 요청 실행 (재시도 로직 포함)"""
        url = urljoin(self.base_url, endpoint)

        for attempt in range(max_retries):
            try:
                logger.debug(f"API 요청: {url}, 파라미터: {params}")
                response = self.session.get(url, params=params, timeout=30)

                if response.status_code == 429:
                    wait_time = min(2 ** attempt, 60)  # 지수 백오프
                    logger.warning(f"Rate limit 도달. {wait_time}초 대기 중...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                logger.error(f"API 요청 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise HuggingFaceAPIError(f"API 요청 최종 실패: {e}")
                time.sleep(delay * (attempt + 1))

        raise HuggingFaceAPIError("모든 재시도 실패")

    def get_models(self, task: Optional[str] = None, limit: int = 100,
                   sort: str = "downloads", author: Optional[str] = None,
                   library: Optional[str] = None) -> List[Dict[str, Any]]:
        """모델 목록을 가져옵니다"""
        params = {
            "limit": min(limit, 1000),  # API 제한
            "sort": sort,
            "direction": -1  # 내림차순
        }

        if task:
            params["pipeline_tag"] = task
        if author:
            params["author"] = author
        if library:
            params["library"] = library

        try:
            logger.info(f"모델 목록 요청 - 태스크: {task}, 제한: {limit}")
            data = self._make_request("/models", params)
            logger.info(f"{len(data)}개 모델 정보를 가져왔습니다.")
            return data
        except Exception as e:
            logger.error(f"모델 목록 가져오기 실패: {e}")
            return []

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """특정 모델의 상세 정보를 가져옵니다"""
        try:
            logger.debug(f"모델 상세 정보 요청: {model_id}")
            data = self._make_request(f"/models/{model_id}")
            return data
        except Exception as e:
            logger.error(f"모델 정보 가져오기 실패 ({model_id}): {e}")
            return None

    def get_model_readme(self, model_id: str) -> Optional[str]:
        """모델의 README 내용을 가져옵니다"""
        try:
            url = f"https://huggingface.co/{model_id}/raw/main/README.md"
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                return response.text
            return None
        except Exception as e:
            logger.error(f"README 가져오기 실패 ({model_id}): {e}")
            return None

    def parse_model_card_evaluations(self, model_id: str) -> List[Dict[str, Any]]:
        """모델 카드에서 평가 결과를 파싱합니다"""
        try:
            model_info = self.get_model_info(model_id)
            if not model_info:
                return []

            evaluations = []

            # model-index에서 평가 결과 추출
            if 'model-index' in model_info:
                for entry in model_info['model-index']:
                    if 'results' in entry:
                        for result in entry['results']:
                            if 'metrics' in result:
                                task_type = result.get('task', {}).get('type', 'unknown')
                                dataset_name = result.get('dataset', {}).get('name', 'unknown')

                                for metric in result['metrics']:
                                    evaluations.append({
                                        'model_id': model_id,
                                        'task_type': task_type,
                                        'dataset_name': dataset_name,
                                        'metric_name': metric.get('name', 'unknown'),
                                        'metric_value': metric.get('value', 0),
                                        'dataset_version': result.get('dataset', {}).get('revision'),
                                        'additional_info': result
                                    })

            # cardData에서도 평가 결과 확인
            if 'cardData' in model_info and 'model-index' in model_info['cardData']:
                for entry in model_info['cardData']['model-index']:
                    if 'results' in entry:
                        for result in entry['results']:
                            if 'metrics' in result:
                                task_type = result.get('task', {}).get('type', 'unknown')
                                dataset_name = result.get('dataset', {}).get('name', 'unknown')

                                for metric in result['metrics']:
                                    evaluations.append({
                                        'model_id': model_id,
                                        'task_type': task_type,
                                        'dataset_name': dataset_name,
                                        'metric_name': metric.get('name', 'unknown'),
                                        'metric_value': metric.get('value', 0),
                                        'dataset_version': result.get('dataset', {}).get('revision'),
                                        'additional_info': result
                                    })

            logger.debug(f"모델 {model_id}에서 {len(evaluations)}개 평가 결과 발견")
            return evaluations

        except Exception as e:
            logger.error(f"평가 결과 파싱 실패 ({model_id}): {e}")
            return []

    def get_model_evaluations(self, model_id: str) -> List[EvaluationResult]:
        """모델의 평가 결과를 EvaluationResult 객체로 반환"""
        evaluations_data = self.parse_model_card_evaluations(model_id)
        evaluations = []

        for eval_data in evaluations_data:
            try:
                evaluation = EvaluationResult(
                    model_id=eval_data['model_id'],
                    dataset_name=eval_data['dataset_name'],
                    metric_name=eval_data['metric_name'],
                    metric_value=float(eval_data['metric_value']),
                    task_type=eval_data['task_type'],
                    dataset_version=eval_data.get('dataset_version'),
                    additional_info=eval_data.get('additional_info')
                )
                evaluations.append(evaluation)
            except (ValueError, TypeError) as e:
                logger.warning(f"평가 결과 변환 실패: {e}")
                continue

        return evaluations

    def search_models(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """모델 검색"""
        params = {
            "search": query,
            "limit": limit,
            "sort": "downloads",
            "direction": -1
        }

        try:
            logger.info(f"모델 검색: '{query}', 제한: {limit}")
            data = self._make_request("/models", params)
            logger.info(f"검색 결과: {len(data)}개 모델")
            return data
        except Exception as e:
            logger.error(f"모델 검색 실패: {e}")
            return []

    def get_trending_models(self, limit: int = 50) -> List[Dict[str, Any]]:
        """인기 상승 모델 목록"""
        return self.get_models(limit=limit, sort="trending")

    def get_recent_models(self, limit: int = 50) -> List[Dict[str, Any]]:
        """최근 업데이트된 모델 목록"""
        return self.get_models(limit=limit, sort="lastModified")

    def convert_to_model_info(self, model_data: Dict[str, Any]) -> ModelInfo:
        """API 응답을 ModelInfo 객체로 변환"""
        return ModelInfo(
            model_id=model_data.get('id', ''),
            model_name=model_data.get('id', '').split('/')[-1],
            author=model_data.get('author', ''),
            downloads=model_data.get('downloads', 0),
            likes=model_data.get('likes', 0),
            created_at=model_data.get('createdAt', ''),
            last_modified=model_data.get('lastModified', ''),
            library_name=model_data.get('library_name', ''),
            pipeline_tag=model_data.get('pipeline_tag', ''),
            tags=model_data.get('tags', []),
            model_size=self._extract_model_size(model_data),
            license=model_data.get('cardData', {}).get('license', '')
        )

    def _extract_model_size(self, model_data: Dict[str, Any]) -> Optional[str]:
        """모델 크기 정보 추출"""
        # tags에서 모델 크기 정보 찾기
        tags = model_data.get('tags', [])
        size_tags = [tag for tag in tags if any(size in tag.lower()
                                                for size in
                                                ['7b', '13b', '30b', '65b', '70b', 'small', 'base', 'large'])]
        return size_tags[0] if size_tags else None

    def close(self):
        """세션 종료"""
        self.session.close()
        logger.info("HuggingFace API 세션이 종료되었습니다.")


# 싱글톤 패턴으로 API 클라이언트 관리
_api_instance: Optional[HuggingFaceAPI] = None


def get_api_client(token: Optional[str] = None) -> HuggingFaceAPI:
    """API 클라이언트 싱글톤 인스턴스 반환"""
    global _api_instance
    if _api_instance is None:
        _api_instance = HuggingFaceAPI(token=token)
    return _api_instance


def close_api_client():
    """API 클라이언트 종료"""
    global _api_instance
    if _api_instance:
        _api_instance.close()
        _api_instance = None