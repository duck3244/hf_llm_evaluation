"""
데이터베이스 관리 모듈 (개선된 버전)
동시성 처리 및 안전한 데이터베이스 연결 관리
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import json
from datetime import datetime
import logging
import threading
import time
from contextlib import contextmanager
import os

from ..models.data_models import (
    ModelInfo, EvaluationResult, TaskCategory, CollectionStats,
    serialize_for_db, deserialize_from_db
)
from ..utils.logger import get_logger, log_database_operation

logger = get_logger(__name__)

# Windows 호환성을 위한 조건부 import
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False


class DatabaseError(Exception):
    """데이터베이스 관련 예외"""
    pass


class DatabaseLockError(DatabaseError):
    """데이터베이스 락 관련 예외"""
    pass


class DatabaseManager:
    """데이터베이스 관리 클래스 (동시성 처리 강화)"""

    def __init__(self, db_path: str = "data/llm_evaluations.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 스레드 로컬 스토리지 (연결 풀)
        self._local = threading.local()

        # 락 파일 경로
        self.lock_file = self.db_path.with_suffix('.lock')

        # 연결 설정
        self.connection_timeout = 30.0
        self.max_retries = 3
        self.retry_delay = 0.1

        logger.info(f"💾 데이터베이스 초기화: {self.db_path}")
        self.init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """스레드 안전한 데이터베이스 연결 획득"""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            try:
                # WAL 모드로 연결 (동시성 개선)
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=self.connection_timeout,
                    check_same_thread=False,
                    isolation_level=None  # autocommit 모드
                )

                # WAL 모드 설정 (Write-Ahead Logging)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")  # 성능과 안전성 균형
                conn.execute("PRAGMA cache_size=10000")     # 캐시 크기 증가
                conn.execute("PRAGMA temp_store=MEMORY")    # 임시 데이터 메모리 저장
                conn.execute("PRAGMA mmap_size=268435456")  # 메모리 맵 크기 (256MB)

                # 외래 키 제약 조건 활성화
                conn.execute("PRAGMA foreign_keys=ON")

                # 연결 검증
                conn.execute("SELECT 1").fetchone()

                self._local.connection = conn
                logger.debug(f"🔗 새 데이터베이스 연결 생성 (스레드: {threading.current_thread().ident})")

            except sqlite3.Error as e:
                logger.error(f"❌ 데이터베이스 연결 실패: {e}")
                raise DatabaseError(f"데이터베이스 연결 실패: {e}")

        return self._local.connection

    @contextmanager
    def _get_connection_with_retry(self):
        """재시도 로직이 포함된 데이터베이스 연결"""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                conn = self._get_connection()

                # 연결 테스트
                conn.execute("BEGIN IMMEDIATE")  # 즉시 배타적 락 획득
                yield conn
                conn.execute("COMMIT")
                return

            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower():
                    last_exception = DatabaseLockError(f"데이터베이스 락 (시도 {attempt + 1}): {e}")
                    logger.warning(f"⚠️  데이터베이스 락 발생, 재시도 중... (시도 {attempt + 1}/{self.max_retries})")

                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))  # 지수 백오프
                        continue
                else:
                    last_exception = DatabaseError(f"데이터베이스 작업 오류: {e}")
                    break

            except Exception as e:
                last_exception = DatabaseError(f"예상치 못한 데이터베이스 오류: {e}")
                break

        # 모든 재시도 실패
        if last_exception:
            raise last_exception

    @contextmanager
    def _file_lock(self):
        """파일 기반 프로세스 간 락"""
        if not HAS_FCNTL:  # Windows에서는 파일 락 건너뛰기
            yield
            return

        lock_fd = None
        try:
            lock_fd = os.open(self.lock_file, os.O_CREAT | os.O_TRUNC | os.O_RDWR)
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        except (OSError, IOError) as e:
            if lock_fd:
                os.close(lock_fd)
            raise DatabaseLockError(f"프로세스 락 획득 실패: {e}")
        finally:
            if lock_fd:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
                    if self.lock_file.exists():
                        self.lock_file.unlink()
                except:
                    pass

    def init_database(self):
        """데이터베이스 및 테이블 초기화 (동시성 안전)"""
        try:
            with self._file_lock():
                with self._get_connection_with_retry() as conn:
                    cursor = conn.cursor()

                    # 모델 정보 테이블
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS models (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            model_id TEXT UNIQUE NOT NULL,
                            model_name TEXT,
                            author TEXT,
                            downloads INTEGER DEFAULT 0,
                            likes INTEGER DEFAULT 0,
                            created_at TEXT,
                            last_modified TEXT,
                            library_name TEXT,
                            pipeline_tag TEXT,
                            tags TEXT,
                            task_categories TEXT,
                            model_size TEXT,
                            license TEXT,
                            collected_at TEXT DEFAULT CURRENT_TIMESTAMP,
                            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                            CONSTRAINT chk_downloads CHECK (downloads >= 0),
                            CONSTRAINT chk_likes CHECK (likes >= 0)
                        )
                    """)

                    # 평가 결과 테이블
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS evaluations (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            model_id TEXT NOT NULL,
                            dataset_name TEXT NOT NULL,
                            metric_name TEXT NOT NULL,
                            metric_value REAL NOT NULL,
                            task_type TEXT,
                            evaluation_date TEXT,
                            dataset_version TEXT,
                            metric_config TEXT,
                            additional_info TEXT,
                            verified BOOLEAN DEFAULT FALSE,
                            collected_at TEXT DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (model_id) REFERENCES models (model_id) ON DELETE CASCADE,
                            CONSTRAINT chk_metric_value CHECK (metric_value IS NOT NULL AND metric_value = metric_value),
                            UNIQUE(model_id, dataset_name, metric_name, dataset_version)
                        )
                    """)

                    # 태스크 카테고리 테이블
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS task_categories (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            task_name TEXT UNIQUE NOT NULL,
                            description TEXT NOT NULL,
                            common_datasets TEXT,
                            common_metrics TEXT,
                            subcategories TEXT,
                            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                            CONSTRAINT chk_task_name CHECK (LENGTH(task_name) > 0),
                            CONSTRAINT chk_description CHECK (LENGTH(description) > 0)
                        )
                    """)

                    # 데이터셋 정보 테이블
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS datasets (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            dataset_name TEXT UNIQUE NOT NULL,
                            task_type TEXT,
                            description TEXT,
                            size INTEGER,
                            language TEXT,
                            splits TEXT,
                            metrics TEXT,
                            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                            CONSTRAINT chk_dataset_name CHECK (LENGTH(dataset_name) > 0),
                            CONSTRAINT chk_size CHECK (size IS NULL OR size >= 0)
                        )
                    """)

                    # 수집 통계 테이블
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS collection_stats (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            collection_date TEXT NOT NULL,
                            total_models INTEGER DEFAULT 0,
                            total_evaluations INTEGER DEFAULT 0,
                            tasks_collected TEXT,
                            collection_duration REAL,
                            errors_count INTEGER DEFAULT 0,
                            success_rate REAL DEFAULT 0.0,
                            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                            CONSTRAINT chk_totals CHECK (
                                total_models >= 0 AND 
                                total_evaluations >= 0 AND 
                                errors_count >= 0 AND
                                success_rate >= 0.0 AND 
                                success_rate <= 1.0
                            )
                        )
                    """)

                    # 성능 최적화를 위한 인덱스 생성
                    indexes = [
                        "CREATE INDEX IF NOT EXISTS idx_models_pipeline_tag ON models (pipeline_tag)",
                        "CREATE INDEX IF NOT EXISTS idx_models_downloads ON models (downloads DESC)",
                        "CREATE INDEX IF NOT EXISTS idx_models_author ON models (author)",
                        "CREATE INDEX IF NOT EXISTS idx_models_updated ON models (updated_at DESC)",
                        "CREATE INDEX IF NOT EXISTS idx_evaluations_model_id ON evaluations (model_id)",
                        "CREATE INDEX IF NOT EXISTS idx_evaluations_task_type ON evaluations (task_type)",
                        "CREATE INDEX IF NOT EXISTS idx_evaluations_metric ON evaluations (metric_name)",
                        "CREATE INDEX IF NOT EXISTS idx_evaluations_value ON evaluations (metric_value DESC)",
                        "CREATE INDEX IF NOT EXISTS idx_evaluations_dataset ON evaluations (dataset_name)",
                        "CREATE INDEX IF NOT EXISTS idx_evaluations_composite ON evaluations (task_type, metric_name, metric_value DESC)"
                    ]

                    for index_sql in indexes:
                        cursor.execute(index_sql)

                    # 트리거 생성 (자동 업데이트)
                    cursor.execute("""
                        CREATE TRIGGER IF NOT EXISTS update_models_timestamp 
                        AFTER UPDATE ON models
                        BEGIN
                            UPDATE models SET updated_at = CURRENT_TIMESTAMP 
                            WHERE id = NEW.id;
                        END
                    """)

                    logger.info("✅ 데이터베이스 테이블, 인덱스 및 트리거 초기화 완료")

        except DatabaseLockError as e:
            logger.error(f"❌ 데이터베이스 초기화 락 오류: {e}")
            raise
        except sqlite3.Error as e:
            logger.error(f"❌ 데이터베이스 초기화 실패: {e}")
            raise DatabaseError(f"데이터베이스 초기화 실패: {e}")

    def insert_model(self, model_info: ModelInfo) -> bool:
        """모델 정보를 데이터베이스에 저장 (동시성 안전)"""
        try:
            with self._get_connection_with_retry() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO models 
                    (model_id, model_name, author, downloads, likes, created_at, 
                     last_modified, library_name, pipeline_tag, tags, task_categories,
                     model_size, license, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_info.model_id,
                    model_info.model_name,
                    model_info.author,
                    model_info.downloads,
                    model_info.likes,
                    model_info.created_at,
                    model_info.last_modified,
                    model_info.library_name,
                    model_info.pipeline_tag,
                    serialize_for_db(model_info.tags),
                    serialize_for_db(model_info.task_categories),
                    model_info.model_size,
                    model_info.license,
                    datetime.now().isoformat()
                ))

                log_database_operation("INSERT", "models", 1, logger)
                logger.debug(f"✅ 모델 저장 완료: {model_info.model_id}")
                return True

        except DatabaseLockError as e:
            logger.error(f"❌ 모델 저장 락 오류 ({model_info.model_id}): {e}")
            return False
        except sqlite3.IntegrityError as e:
            logger.warning(f"⚠️  모델 중복 또는 제약 조건 위반 ({model_info.model_id}): {e}")
            return False
        except sqlite3.Error as e:
            logger.error(f"❌ 모델 저장 실패 ({model_info.model_id}): {e}")
            return False

    def insert_evaluation(self, evaluation: EvaluationResult) -> bool:
        """평가 결과를 데이터베이스에 저장 (동시성 안전)"""
        try:
            with self._get_connection_with_retry() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO evaluations 
                    (model_id, dataset_name, metric_name, metric_value, 
                     task_type, evaluation_date, dataset_version, metric_config,
                     additional_info, verified)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    evaluation.model_id,
                    evaluation.dataset_name,
                    evaluation.metric_name,
                    evaluation.metric_value,
                    evaluation.task_type,
                    evaluation.evaluation_date,
                    evaluation.dataset_version,
                    serialize_for_db(evaluation.metric_config),
                    serialize_for_db(evaluation.additional_info),
                    evaluation.verified
                ))

                log_database_operation("INSERT", "evaluations", 1, logger)
                logger.debug(f"✅ 평가 결과 저장: {evaluation.model_id} - {evaluation.metric_name}")
                return True

        except DatabaseLockError as e:
            logger.error(f"❌ 평가 결과 저장 락 오류: {e}")
            return False
        except sqlite3.IntegrityError as e:
            logger.warning(f"⚠️  평가 결과 중복 또는 제약 조건 위반: {e}")
            return False
        except sqlite3.Error as e:
            logger.error(f"❌ 평가 결과 저장 실패: {e}")
            return False

    def insert_task_category(self, task_category: TaskCategory) -> bool:
        """태스크 카테고리 정보 저장 (동시성 안전)"""
        try:
            with self._get_connection_with_retry() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO task_categories 
                    (task_name, description, common_datasets, common_metrics, subcategories)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    task_category.task_name,
                    task_category.description,
                    serialize_for_db(task_category.common_datasets),
                    serialize_for_db(task_category.common_metrics),
                    serialize_for_db(task_category.subcategories)
                ))

                log_database_operation("INSERT", "task_categories", 1, logger)
                return True

        except DatabaseLockError as e:
            logger.error(f"❌ 태스크 카테고리 저장 락 오류: {e}")
            return False
        except sqlite3.Error as e:
            logger.error(f"❌ 태스크 카테고리 저장 실패: {e}")
            return False

    def get_models_by_task(self, task: str, limit: Optional[int] = None) -> pd.DataFrame:
        """특정 태스크의 모델들을 조회 (동시성 안전)"""
        try:
            with self._get_connection_with_retry() as conn:
                query = """
                    SELECT * FROM models 
                    WHERE pipeline_tag = ? OR task_categories LIKE ?
                    ORDER BY downloads DESC
                """
                params = [task, f'%{task}%']

                if limit:
                    query += " LIMIT ?"
                    params.append(limit)

                df = pd.read_sql_query(query, conn, params=params)

                # JSON 컬럼 역직렬화
                if not df.empty:
                    df['tags'] = df['tags'].apply(lambda x: deserialize_from_db(x, list) if x else [])
                    df['task_categories'] = df['task_categories'].apply(
                        lambda x: deserialize_from_db(x, list) if x else [])

                logger.debug(f"📊 태스크 '{task}' 모델 조회: {len(df)}개")
                return df

        except DatabaseLockError as e:
            logger.error(f"❌ 모델 조회 락 오류: {e}")
            return pd.DataFrame()
        except sqlite3.Error as e:
            logger.error(f"❌ 모델 조회 실패: {e}")
            return pd.DataFrame()

    def get_evaluations_by_model(self, model_id: str) -> pd.DataFrame:
        """특정 모델의 평가 결과를 조회 (동시성 안전)"""
        try:
            with self._get_connection_with_retry() as conn:
                query = "SELECT * FROM evaluations WHERE model_id = ? ORDER BY collected_at DESC"
                df = pd.read_sql_query(query, conn, params=[model_id])

                # JSON 컬럼 역직렬화
                if not df.empty:
                    df['additional_info'] = df['additional_info'].apply(
                        lambda x: deserialize_from_db(x, dict) if x else {}
                    )
                    df['metric_config'] = df['metric_config'].apply(
                        lambda x: deserialize_from_db(x, dict) if x else {}
                    )

                logger.debug(f"📊 모델 '{model_id}' 평가 결과 조회: {len(df)}개")
                return df

        except DatabaseLockError as e:
            logger.error(f"❌ 평가 결과 조회 락 오류: {e}")
            return pd.DataFrame()
        except sqlite3.Error as e:
            logger.error(f"❌ 평가 결과 조회 실패: {e}")
            return pd.DataFrame()

    def get_task_leaderboard(self, task_type: str, metric_name: str,
                             dataset_name: Optional[str] = None, limit: int = 50) -> pd.DataFrame:
        """태스크별 리더보드를 생성 (동시성 안전)"""
        try:
            with self._get_connection_with_retry() as conn:
                query = """
                    SELECT m.model_id, m.model_name, m.author, e.metric_value, 
                           e.dataset_name, e.evaluation_date, m.downloads, m.likes,
                           m.model_size, m.license, e.verified
                    FROM models m
                    JOIN evaluations e ON m.model_id = e.model_id
                    WHERE e.task_type = ? AND e.metric_name = ?
                """
                params = [task_type, metric_name]

                if dataset_name:
                    query += " AND e.dataset_name = ?"
                    params.append(dataset_name)

                query += " ORDER BY e.metric_value DESC LIMIT ?"
                params.append(limit)

                df = pd.read_sql_query(query, conn, params=params)
                logger.debug(f"🏆 리더보드 생성: {task_type}-{metric_name} ({len(df)}개)")
                return df

        except DatabaseLockError as e:
            logger.error(f"❌ 리더보드 조회 락 오류: {e}")
            return pd.DataFrame()
        except sqlite3.Error as e:
            logger.error(f"❌ 리더보드 조회 실패: {e}")
            return pd.DataFrame()

    def get_model_statistics(self) -> Dict[str, Any]:
        """모델 통계 정보 반환 (동시성 안전)"""
        try:
            with self._get_connection_with_retry() as conn:
                cursor = conn.cursor()

                # 전체 모델 수
                cursor.execute("SELECT COUNT(*) FROM models")
                total_models = cursor.fetchone()[0]

                # 태스크별 모델 수
                cursor.execute("""
                    SELECT pipeline_tag, COUNT(*) 
                    FROM models 
                    WHERE pipeline_tag IS NOT NULL AND pipeline_tag != ''
                    GROUP BY pipeline_tag 
                    ORDER BY COUNT(*) DESC
                """)
                task_counts = dict(cursor.fetchall())

                # 평가 결과 수
                cursor.execute("SELECT COUNT(*) FROM evaluations")
                total_evaluations = cursor.fetchone()[0]

                # 최근 수집 날짜
                cursor.execute("SELECT MAX(collected_at) FROM models")
                last_collection = cursor.fetchone()[0]

                # 상위 작성자
                cursor.execute("""
                    SELECT author, COUNT(*) as model_count
                    FROM models 
                    WHERE author IS NOT NULL AND author != ''
                    GROUP BY author 
                    ORDER BY model_count DESC 
                    LIMIT 5
                """)
                top_authors = dict(cursor.fetchall())

                stats = {
                    'total_models': total_models,
                    'total_evaluations': total_evaluations,
                    'task_counts': task_counts,
                    'last_collection': last_collection,
                    'top_authors': top_authors
                }

                logger.debug(f"📊 통계 조회 완료: {total_models:,}개 모델, {total_evaluations:,}개 평가")
                return stats

        except DatabaseLockError as e:
            logger.error(f"❌ 통계 조회 락 오류: {e}")
            return {}
        except sqlite3.Error as e:
            logger.error(f"❌ 통계 조회 실패: {e}")
            return {}

    def search_models(self, query: str, limit: int = 50) -> pd.DataFrame:
        """모델 검색 (동시성 안전)"""
        try:
            with self._get_connection_with_retry() as conn:
                sql_query = """
                    SELECT * FROM models 
                    WHERE model_id LIKE ? OR model_name LIKE ? OR author LIKE ?
                    ORDER BY downloads DESC
                    LIMIT ?
                """
                search_term = f"%{query}%"
                params = [search_term, search_term, search_term, limit]

                df = pd.read_sql_query(sql_query, conn, params=params)
                logger.debug(f"🔍 모델 검색 '{query}': {len(df)}개 결과")
                return df

        except DatabaseLockError as e:
            logger.error(f"❌ 모델 검색 락 오류: {e}")
            return pd.DataFrame()
        except sqlite3.Error as e:
            logger.error(f"❌ 모델 검색 실패: {e}")
            return pd.DataFrame()

    def get_top_models_by_downloads(self, limit: int = 50, task: Optional[str] = None) -> pd.DataFrame:
        """다운로드 수 기준 상위 모델 (동시성 안전)"""
        try:
            with self._get_connection_with_retry() as conn:
                query = "SELECT * FROM models"
                params = []

                if task:
                    query += " WHERE pipeline_tag = ?"
                    params.append(task)

                query += " ORDER BY downloads DESC LIMIT ?"
                params.append(limit)

                df = pd.read_sql_query(query, conn, params=params)
                logger.debug(f"🏆 상위 모델 조회: {len(df)}개")
                return df

        except DatabaseLockError as e:
            logger.error(f"❌ 상위 모델 조회 락 오류: {e}")
            return pd.DataFrame()
        except sqlite3.Error as e:
            logger.error(f"❌ 상위 모델 조회 실패: {e}")
            return pd.DataFrame()

    def export_to_csv(self, output_dir: str = "exports"):
        """데이터를 CSV로 내보내기 (동시성 안전)"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            exported_files = []

            with self._get_connection_with_retry() as conn:
                # 모델 데이터 내보내기
                models_df = pd.read_sql_query("SELECT * FROM models ORDER BY downloads DESC", conn)
                if not models_df.empty:
                    models_file = output_path / "models.csv"
                    models_df.to_csv(models_file, index=False, encoding='utf-8')
                    exported_files.append(str(models_file))

                # 평가 데이터 내보내기
                evaluations_df = pd.read_sql_query("SELECT * FROM evaluations ORDER BY collected_at DESC", conn)
                if not evaluations_df.empty:
                    evaluations_file = output_path / "evaluations.csv"
                    evaluations_df.to_csv(evaluations_file, index=False, encoding='utf-8')
                    exported_files.append(str(evaluations_file))

                # 태스크 카테고리 내보내기
                tasks_df = pd.read_sql_query("SELECT * FROM task_categories", conn)
                if not tasks_df.empty:
                    tasks_file = output_path / "task_categories.csv"
                    tasks_df.to_csv(tasks_file, index=False, encoding='utf-8')
                    exported_files.append(str(tasks_file))

                # 수집 통계 내보내기
                stats_df = pd.read_sql_query("SELECT * FROM collection_stats ORDER BY created_at DESC", conn)
                if not stats_df.empty:
                    stats_file = output_path / "collection_stats.csv"
                    stats_df.to_csv(stats_file, index=False, encoding='utf-8')
                    exported_files.append(str(stats_file))

            logger.info(f"📁 데이터 내보내기 완료: {len(exported_files)}개 파일")
            for file_path in exported_files:
                logger.info(f"   • {file_path}")

        except DatabaseLockError as e:
            logger.error(f"❌ 데이터 내보내기 락 오류: {e}")
        except Exception as e:
            logger.error(f"❌ 데이터 내보내기 실패: {e}")

    def backup_database(self, backup_path: Optional[str] = None):
        """데이터베이스 백업 (동시성 안전)"""
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"data/backup_llm_evaluations_{timestamp}.db"

        try:
            backup_path_obj = Path(backup_path)
            backup_path_obj.parent.mkdir(parents=True, exist_ok=True)

            with self._get_connection_with_retry() as conn:
                # SQLite 백업 API 사용 (원자적 백업)
                backup_conn = sqlite3.connect(backup_path)
                try:
                    conn.backup(backup_conn)
                    logger.info(f"💾 데이터베이스 백업 완료: {backup_path}")
                    return backup_path
                finally:
                    backup_conn.close()

        except DatabaseLockError as e:
            logger.error(f"❌ 데이터베이스 백업 락 오류: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ 데이터베이스 백업 실패: {e}")
            return None

    def vacuum_database(self):
        """데이터베이스 최적화 (동시성 안전)"""
        try:
            with self._file_lock():  # 전체 데이터베이스 락 필요
                with self._get_connection_with_retry() as conn:
                    logger.info("🔧 데이터베이스 최적화 시작...")

                    # WAL 체크포인트
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

                    # VACUUM 실행
                    conn.execute("VACUUM")

                    # 통계 업데이트
                    conn.execute("ANALYZE")

                    logger.info("✅ 데이터베이스 최적화 완료")

        except DatabaseLockError as e:
            logger.error(f"❌ 데이터베이스 최적화 락 오류: {e}")
        except sqlite3.Error as e:
            logger.error(f"❌ 데이터베이스 최적화 실패: {e}")

    def get_database_size(self) -> Dict[str, Any]:
        """데이터베이스 크기 정보 조회"""
        try:
            file_size = self.db_path.stat().st_size
            size_mb = file_size / (1024 * 1024)

            with self._get_connection_with_retry() as conn:
                cursor = conn.cursor()

                # 테이블별 레코드 수
                tables = ['models', 'evaluations', 'task_categories', 'collection_stats']
                table_counts = {}

                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    table_counts[table] = cursor.fetchone()[0]

                # WAL 파일 크기
                wal_file = self.db_path.with_suffix('.db-wal')
                wal_size = 0
                if wal_file.exists():
                    wal_size = wal_file.stat().st_size

            return {
                'file_size_bytes': file_size,
                'file_size_mb': round(size_mb, 2),
                'wal_size_bytes': wal_size,
                'wal_size_mb': round(wal_size / (1024 * 1024), 2),
                'table_counts': table_counts,
                'db_path': str(self.db_path)
            }

        except Exception as e:
            logger.error(f"❌ 데이터베이스 크기 조회 실패: {e}")
            return {}

    def health_check(self) -> Dict[str, Any]:
        """데이터베이스 상태 점검"""
        try:
            start_time = time.time()

            with self._get_connection_with_retry() as conn:
                cursor = conn.cursor()

                # 연결 테스트
                cursor.execute("SELECT 1")

                # 무결성 검사 (빠른 검사)
                cursor.execute("PRAGMA quick_check")
                integrity_result = cursor.fetchone()[0]

                # WAL 모드 확인
                cursor.execute("PRAGMA journal_mode")
                journal_mode = cursor.fetchone()[0]

                # 외래 키 제약 조건 확인
                cursor.execute("PRAGMA foreign_keys")
                foreign_keys = cursor.fetchone()[0]

            response_time = time.time() - start_time

            health = {
                'status': 'healthy' if integrity_result == 'ok' else 'unhealthy',
                'response_time': round(response_time, 3),
                'integrity_check': integrity_result,
                'journal_mode': journal_mode,
                'foreign_keys_enabled': bool(foreign_keys),
                'connection_timeout': self.connection_timeout,
                'wal_enabled': journal_mode.upper() == 'WAL'
            }

            if health['status'] == 'healthy':
                logger.debug("✅ 데이터베이스 상태 정상")
            else:
                logger.warning(f"⚠️  데이터베이스 상태 이상: {integrity_result}")

            return health

        except DatabaseLockError as e:
            logger.error(f"❌ 데이터베이스 상태 점검 락 오류: {e}")
            return {'status': 'error', 'error': str(e)}
        except Exception as e:
            logger.error(f"❌ 데이터베이스 상태 점검 실패: {e}")
            return {'status': 'error', 'error': str(e)}

    def get_connection_info(self) -> Dict[str, Any]:
        """현재 연결 정보 반환"""
        try:
            thread_id = threading.current_thread().ident
            has_connection = hasattr(self._local, 'connection') and self._local.connection is not None

            info = {
                'thread_id': thread_id,
                'has_local_connection': has_connection,
                'db_path': str(self.db_path),
                'connection_timeout': self.connection_timeout,
                'max_retries': self.max_retries,
                'lock_file_exists': self.lock_file.exists()
            }

            if has_connection:
                try:
                    # 연결 상태 확인
                    self._local.connection.execute("SELECT 1")
                    info['connection_status'] = 'active'
                except:
                    info['connection_status'] = 'inactive'
            else:
                info['connection_status'] = 'none'

            return info

        except Exception as e:
            logger.error(f"❌ 연결 정보 조회 실패: {e}")
            return {'error': str(e)}

    def close_connection(self):
        """현재 스레드의 데이터베이스 연결 종료"""
        if hasattr(self._local, 'connection') and self._local.connection is not None:
            try:
                self._local.connection.close()
                self._local.connection = None
                logger.debug(f"🔒 데이터베이스 연결 종료 (스레드: {threading.current_thread().ident})")
            except Exception as e:
                logger.error(f"❌ 연결 종료 중 오류: {e}")

    def close(self):
        """데이터베이스 관리자 종료"""
        try:
            self.close_connection()

            # 락 파일 정리
            if self.lock_file.exists():
                try:
                    self.lock_file.unlink()
                except:
                    pass

            logger.debug("🔒 데이터베이스 관리자 종료")
        except Exception as e:
            logger.error(f"❌ 데이터베이스 관리자 종료 중 오류: {e}")

    def __enter__(self):
        """컨텍스트 매니저 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.close()

    def __del__(self):
        """소멸자"""
        try:
            self.close()
        except:
            pass