"""
HuggingFace API 클라이언트 모듈 (개선된 버전)
강화된 오류 처리 및 API 제한 관리
"""

import requests
import time
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urljoin
import logging
from enum import Enum
import json

from ..models.data_models import ModelInfo, EvaluationResult
from ..utils.logger import get_logger, log_api_request, log_execution_time

logger = get_logger(__name__)


class APIErrorType(Enum):
    """API 오류 타입 분류"""
    RATE_LIMIT = "rate_limit"
    AUTH_ERROR = "auth_error"
    SERVER_ERROR = "server_error"
    CLIENT_ERROR = "client_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"


class HuggingFaceAPIError(Exception):
    """HuggingFace API 관련 예외"""
    def __init__(self, message: str, error_type: APIErrorType = APIErrorType.CLIENT_ERROR,
                 status_code: Optional[int] = None):
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code


class RateLimitError(HuggingFaceAPIError):
    """API 요청 제한 예외"""
    def __init__(self, message: str = "API 요청 제한에 도달했습니다", retry_after: Optional[int] = None):
        super().__init__(message, APIErrorType.RATE_LIMIT, 429)
        self.retry_after = retry_after


class AuthenticationError(HuggingFaceAPIError):
    """인증 오류 예외"""
    def __init__(self, message: str = "API 토큰이 유효하지 않거나 권한이 없습니다"):
        super().__init__(message, APIErrorType.AUTH_ERROR, 403)


class ServerError(HuggingFaceAPIError):
    """서버 오류 예외"""
    def __init__(self, message: str, status_code: int):
        super().__init__(message, APIErrorType.SERVER_ERROR, status_code)


class HuggingFaceAPI:
    """HuggingFace API 클라이언트 (강화된 오류 처리)"""

    def __init__(self, token: Optional[str] = None, base_url: str = "https://huggingface.co/api"):
        self.base_url = base_url
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "HF-LLM-Evaluation-Collector/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        if token:
            self.headers["Authorization"] = f"Bearer {token}"
            logger.info("✅ HuggingFace API 토큰이 설정되었습니다.")
        else:
            logger.warning("⚠️  HuggingFace API 토큰이 설정되지 않았습니다. 일부 기능이 제한될 수 있습니다.")

        self.session.headers.update(self.headers)

        # 연결 풀 설정
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=0  # 직접 재시도 로직 구현
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    @log_execution_time
    def _make_request(self, endpoint: str, params: Optional[Dict] = None,
                      max_retries: int = 3, delay: float = 0.1) -> Dict[str, Any]:
        """강화된 API 요청 실행"""
        url = urljoin(self.base_url, endpoint)
        log_api_request(endpoint, params or {}, logger)

        last_exception = None

        for attempt in range(max_retries):
            try:
                logger.debug(f"🌐 API 요청 (시도 {attempt + 1}/{max_retries}): {url}")

                response = self.session.get(
                    url,
                    params=params,
                    timeout=(10, 30)  # (연결 타임아웃, 읽기 타임아웃)
                )

                # 상태 코드별 처리
                if response.status_code == 200:
                    try:
                        data = response.json()
                        logger.debug(f"✅ API 요청 성공: {len(data) if isinstance(data, list) else 1}개 항목")
                        return data
                    except json.JSONDecodeError as e:
                        raise HuggingFaceAPIError(
                            f"잘못된 JSON 응답: {e}",
                            APIErrorType.CLIENT_ERROR
                        )

                elif response.status_code == 429:
                    # Rate limit 처리
                    retry_after = self._get_retry_after(response)
                    wait_time = min(retry_after or (2 ** attempt), 300)  # 최대 5분

                    logger.warning(f"⏰ Rate limit 도달. {wait_time}초 대기 중... (시도 {attempt + 1})")

                    if attempt == max_retries - 1:
                        raise RateLimitError(
                            f"Rate limit 초과 (최대 재시도 {max_retries}회 도달)",
                            retry_after
                        )

                    time.sleep(wait_time)
                    continue

                elif response.status_code == 401:
                    raise AuthenticationError("API 토큰이 유효하지 않습니다")

                elif response.status_code == 403:
                    raise AuthenticationError("API 접근 권한이 없습니다")

                elif response.status_code == 404:
                    raise HuggingFaceAPIError(
                        f"요청한 리소스를 찾을 수 없습니다: {endpoint}",
                        APIErrorType.CLIENT_ERROR,
                        404
                    )

                elif 500 <= response.status_code < 600:
                    # 서버 오류 처리
                    wait_time = min(1.5 ** attempt, 60)  # 지수 백오프, 최대 1분

                    logger.warning(f"🔧 서버 오류 {response.status_code}. {wait_time}초 후 재시도... (시도 {attempt + 1})")

                    if attempt == max_retries - 1:
                        raise ServerError(
                            f"서버 오류: HTTP {response.status_code}",
                            response.status_code
                        )

                    time.sleep(wait_time)
                    continue

                else:
                    # 기타 클라이언트 오류
                    raise HuggingFaceAPIError(
                        f"HTTP {response.status_code}: {response.text[:200]}",
                        APIErrorType.CLIENT_ERROR,
                        response.status_code
                    )

            except requests.exceptions.Timeout as e:
                logger.warning(f"⏰ API 요청 타임아웃 (시도 {attempt + 1}/{max_retries})")
                last_exception = HuggingFaceAPIError(
                    f"API 요청 타임아웃: {e}",
                    APIErrorType.TIMEOUT_ERROR
                )

                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 1))
                    continue

            except requests.exceptions.ConnectionError as e:
                logger.warning(f"🌐 네트워크 연결 오류 (시도 {attempt + 1}/{max_retries})")
                last_exception = HuggingFaceAPIError(
                    f"네트워크 연결 오류: {e}",
                    APIErrorType.NETWORK_ERROR
                )

                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 1))
                    continue

            except requests.exceptions.RequestException as e:
                logger.error(f"❌ API 요청 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                last_exception = HuggingFaceAPIError(
                    f"API 요청 실패: {e}",
                    APIErrorType.CLIENT_ERROR
                )

                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 1))
                    continue

        # 모든 재시도 실패
        if last_exception:
            raise last_exception
        else:
            raise HuggingFaceAPIError("모든 재시도 실패")

    def _get_retry_after(self, response: requests.Response) -> Optional[int]:
        """Retry-After 헤더에서 대기 시간 추출"""
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            try:
                return int(retry_after)
            except ValueError:
                pass

        # X-RateLimit-Reset 헤더 확인
        rate_limit_reset = response.headers.get('X-RateLimit-Reset')
        if rate_limit_reset:
            try:
                import time as time_module
                reset_time = int(rate_limit_reset)
                current_time = int(time_module.time())
                return max(0, reset_time - current_time)
            except ValueError:
                pass

        return None

    def get_models(self, task: Optional[str] = None, limit: int = 100,
                   sort: str = "downloads", author: Optional[str] = None,
                   library: Optional[str] = None) -> List[Dict[str, Any]]:
        """모델 목록을 가져옵니다 (강화된 오류 처리)"""
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
            logger.info(f"📥 모델 목록 요청 - 태스크: {task or 'ALL'}, 제한: {limit}")
            data = self._make_request("/models", params)

            if isinstance(data, list):
                logger.info(f"✅ {len(data):,}개 모델 정보를 가져왔습니다.")
                return data
            else:
                logger.warning("예상과 다른 응답 형식입니다.")
                return []

        except (RateLimitError, AuthenticationError) as e:
            logger.error(f"❌ API 오류: {e}")
            raise
        except HuggingFaceAPIError as e:
            logger.error(f"❌ HuggingFace API 오류: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ 예상치 못한 오류: {e}")
            raise HuggingFaceAPIError(f"예상치 못한 오류: {e}")

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """특정 모델의 상세 정보를 가져옵니다 (강화된 오류 처리)"""
        try:
            logger.debug(f"🔍 모델 상세 정보 요청: {model_id}")
            data = self._make_request(f"/models/{model_id}")
            logger.debug(f"✅ 모델 정보 수집 완료: {model_id}")
            return data

        except AuthenticationError as e:
            logger.error(f"❌ 인증 오류 ({model_id}): {e}")
            raise
        except HuggingFaceAPIError as e:
            if e.status_code == 404:
                logger.warning(f"⚠️  모델을 찾을 수 없음: {model_id}")
                return None
            logger.error(f"❌ API 오류 ({model_id}): {e}")
            return None
        except Exception as e:
            logger.error(f"❌ 예상치 못한 오류 ({model_id}): {e}")
            return None

    def get_model_readme(self, model_id: str) -> Optional[str]:
        """모델의 README 내용을 가져옵니다"""
        try:
            url = f"https://huggingface.co/{model_id}/raw/main/README.md"
            response = self.session.get(url, timeout=(10, 30))

            if response.status_code == 200:
                logger.debug(f"✅ README 수집 완료: {model_id}")
                return response.text
            elif response.status_code == 404:
                logger.debug(f"📄 README 없음: {model_id}")
                return None
            else:
                logger.warning(f"⚠️  README 수집 실패 ({model_id}): HTTP {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            logger.warning(f"⏰ README 요청 타임아웃: {model_id}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ README 가져오기 실패 ({model_id}): {e}")
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
                evaluations.extend(self._extract_evaluations_from_model_index(
                    model_info['model-index'], model_id
                ))

            # cardData에서도 평가 결과 확인
            if 'cardData' in model_info and 'model-index' in model_info['cardData']:
                evaluations.extend(self._extract_evaluations_from_model_index(
                    model_info['cardData']['model-index'], model_id
                ))

            logger.debug(f"📊 모델 {model_id}에서 {len(evaluations)}개 평가 결과 발견")
            return evaluations

        except Exception as e:
            logger.error(f"❌ 평가 결과 파싱 실패 ({model_id}): {e}")
            return []

    def _extract_evaluations_from_model_index(self, model_index: List[Dict], model_id: str) -> List[Dict[str, Any]]:
        """model-index에서 평가 결과 추출"""
        evaluations = []

        try:
            for entry in model_index:
                if 'results' not in entry:
                    continue

                for result in entry['results']:
                    if 'metrics' not in result:
                        continue

                    task_type = result.get('task', {}).get('type', 'unknown')
                    dataset_name = result.get('dataset', {}).get('name', 'unknown')

                    for metric in result['metrics']:
                        evaluation = {
                            'model_id': model_id,
                            'task_type': task_type,
                            'dataset_name': dataset_name,
                            'metric_name': metric.get('name', 'unknown'),
                            'metric_value': metric.get('value', 0),
                            'dataset_version': result.get('dataset', {}).get('revision'),
                            'additional_info': result
                        }
                        evaluations.append(evaluation)
        except Exception as e:
            logger.error(f"❌ model-index 파싱 오류: {e}")

        return evaluations

    def get_model_evaluations(self, model_id: str) -> List[EvaluationResult]:
        """모델의 평가 결과를 EvaluationResult 객체로 반환"""
        evaluations_data = self.parse_model_card_evaluations(model_id)
        evaluations = []

        for eval_data in evaluations_data:
            try:
                # 메트릭 값 검증
                metric_value = eval_data['metric_value']
                if not isinstance(metric_value, (int, float)):
                    try:
                        metric_value = float(metric_value)
                    except (ValueError, TypeError):
                        logger.warning(f"잘못된 메트릭 값: {metric_value}")
                        continue

                # NaN 값 체크
                if metric_value != metric_value:  # NaN 체크
                    logger.warning(f"NaN 메트릭 값 발견: {eval_data}")
                    continue

                evaluation = EvaluationResult(
                    model_id=eval_data['model_id'],
                    dataset_name=eval_data['dataset_name'],
                    metric_name=eval_data['metric_name'],
                    metric_value=metric_value,
                    task_type=eval_data['task_type'],
                    dataset_version=eval_data.get('dataset_version'),
                    additional_info=eval_data.get('additional_info')
                )
                evaluations.append(evaluation)

            except (ValueError, TypeError) as e:
                logger.warning(f"⚠️  평가 결과 변환 실패: {e}")
                continue
            except Exception as e:
                logger.error(f"❌ 예상치 못한 오류: {e}")
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
            logger.info(f"🔍 모델 검색: '{query}', 제한: {limit}")
            data = self._make_request("/models", params)
            logger.info(f"✅ 검색 결과: {len(data) if isinstance(data, list) else 0}개 모델")
            return data if isinstance(data, list) else []
        except HuggingFaceAPIError as e:
            logger.error(f"❌ 모델 검색 실패: {e}")
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
        size_patterns = ['7b', '13b', '30b', '65b', '70b', '175b', 'small', 'base', 'large', 'xl']

        for tag in tags:
            tag_lower = tag.lower()
            for pattern in size_patterns:
                if pattern in tag_lower:
                    return tag

        # 모델 이름에서 크기 정보 추출
        model_id = model_data.get('id', '').lower()
        for pattern in size_patterns:
            if pattern in model_id:
                return pattern.upper()

        return None

    def check_api_status(self) -> Dict[str, Any]:
        """API 상태 확인"""
        try:
            # 간단한 요청으로 API 상태 확인
            start_time = time.time()
            response = self.session.get(
                f"{self.base_url}/models",
                params={"limit": 1},
                timeout=(5, 10)
            )
            response_time = time.time() - start_time

            status = {
                "available": response.status_code == 200,
                "status_code": response.status_code,
                "response_time": response_time,
                "has_token": "Authorization" in self.headers,
                "api_version": response.headers.get("X-API-Version", "unknown")
            }

            if status["available"]:
                logger.info("✅ HuggingFace API 연결 정상")
            else:
                logger.warning(f"⚠️  HuggingFace API 상태 이상: {response.status_code}")

            return status

        except Exception as e:
            logger.error(f"❌ API 상태 확인 실패: {e}")
            return {
                "available": False,
                "error": str(e),
                "has_token": "Authorization" in self.headers
            }

    def close(self):
        """세션 종료"""
        try:
            self.session.close()
            logger.info("🔒 HuggingFace API 세션이 종료되었습니다.")
        except Exception as e:
            logger.error(f"❌ API 세션 종료 중 오류: {e}")


# 싱글톤 패턴으로 API 클라이언트 관리
_api_instance: Optional[HuggingFaceAPI] = None


def get_api_client(token: Optional[str] = None) -> HuggingFaceAPI:
    """API 클라이언트 싱글톤 인스턴스 반환"""
    global _api_instance
    if _api_instance is None:
        _api_instance = HuggingFaceAPI(token=token)

        # API 상태 확인
        status = _api_instance.check_api_status()
        if not status.get("available", False):
            logger.warning("⚠️  HuggingFace API에 연결할 수 없습니다. 기능이 제한될 수 있습니다.")

    return _api_instance


def close_api_client():
    """API 클라이언트 종료"""
    global _api_instance
    if _api_instance:
        _api_instance.close()
        _api_instance = None