"""
데이터베이스 관리 모듈
README 예시와 일치하도록 수정된 SQLite 데이터 저장 및 조회 기능
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import json
from datetime import datetime
import logging

from ..models.data_models import (
    ModelInfo, EvaluationResult, TaskCategory, CollectionStats,
    serialize_for_db, deserialize_from_db
)
from ..utils.logger import get_logger, log_database_operation

logger = get_logger(__name__)


class DatabaseError(Exception):
    """데이터베이스 관련 예외"""
    pass


class DatabaseManager:
    """데이터베이스 관리 클래스 (README 예시 구현)"""

    def __init__(self, db_path: str = "data/llm_evaluations.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"💾 데이터베이스 초기화: {self.db_path}")
        self.init_database()

    def init_database(self):
        """데이터베이스 및 테이블 초기화 (README 데이터베이스 스키마 구현)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 모델 정보 테이블 (README 모델 정보 구현)
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
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # 평가 결과 테이블 (README 평가 데이터 구현)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS evaluations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id TEXT NOT NULL,
                        dataset_name TEXT,
                        metric_name TEXT,
                        metric_value REAL,
                        task_type TEXT,
                        evaluation_date TEXT,
                        dataset_version TEXT,
                        metric_config TEXT,
                        additional_info TEXT,
                        verified BOOLEAN DEFAULT FALSE,
                        collected_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (model_id) REFERENCES models (model_id)
                    )
                """)

                # 태스크 카테고리 테이블 (README 태스크 테이블 구현)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS task_categories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_name TEXT UNIQUE NOT NULL,
                        description TEXT,
                        common_datasets TEXT,
                        common_metrics TEXT,
                        subcategories TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
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
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # 수집 통계 테이블 (README 통계 구현)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS collection_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        collection_date TEXT,
                        total_models INTEGER DEFAULT 0,
                        total_evaluations INTEGER DEFAULT 0,
                        tasks_collected TEXT,
                        collection_duration REAL,
                        errors_count INTEGER DEFAULT 0,
                        success_rate REAL DEFAULT 0.0,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # 성능 최적화를 위한 인덱스 생성
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_models_pipeline_tag ON models (pipeline_tag)",
                    "CREATE INDEX IF NOT EXISTS idx_models_downloads ON models (downloads DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_models_author ON models (author)",
                    "CREATE INDEX IF NOT EXISTS idx_evaluations_model_id ON evaluations (model_id)",
                    "CREATE INDEX IF NOT EXISTS idx_evaluations_task_type ON evaluations (task_type)",
                    "CREATE INDEX IF NOT EXISTS idx_evaluations_metric ON evaluations (metric_name)",
                    "CREATE INDEX IF NOT EXISTS idx_evaluations_value ON evaluations (metric_value DESC)"
                ]

                for index_sql in indexes:
                    cursor.execute(index_sql)

                conn.commit()
                logger.info("✅ 데이터베이스 테이블 및 인덱스 초기화 완료")

        except sqlite3.Error as e:
            logger.error(f"❌ 데이터베이스 초기화 실패: {e}")
            raise DatabaseError(f"데이터베이스 초기화 실패: {e}")

    def insert_model(self, model_info: ModelInfo) -> bool:
        """모델 정보를 데이터베이스에 저장 (README 예시 구현)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
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

                conn.commit()
                log_database_operation("INSERT", "models", 1, logger)
                logger.debug(f"✅ 모델 저장 완료: {model_info.model_id}")
                return True

        except sqlite3.Error as e:
            logger.error(f"❌ 모델 저장 실패 ({model_info.model_id}): {e}")
            return False

    def insert_evaluation(self, evaluation: EvaluationResult) -> bool:
        """평가 결과를 데이터베이스에 저장 (README 예시 구현)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO evaluations 
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

                conn.commit()
                log_database_operation("INSERT", "evaluations", 1, logger)
                logger.debug(f"✅ 평가 결과 저장: {evaluation.model_id} - {evaluation.metric_name}")
                return True

        except sqlite3.Error as e:
            logger.error(f"❌ 평가 결과 저장 실패: {e}")
            return False

    def insert_task_category(self, task_category: TaskCategory) -> bool:
        """태스크 카테고리 정보 저장 (README 태스크 테이블 구현)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
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

                conn.commit()
                log_database_operation("INSERT", "task_categories", 1, logger)
                return True

        except sqlite3.Error as e:
            logger.error(f"❌ 태스크 카테고리 저장 실패: {e}")
            return False

    def get_models_by_task(self, task: str, limit: Optional[int] = None) -> pd.DataFrame:
        """특정 태스크의 모델들을 조회 (README 예시 구현)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
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

        except sqlite3.Error as e:
            logger.error(f"❌ 모델 조회 실패: {e}")
            return pd.DataFrame()

    def get_evaluations_by_model(self, model_id: str) -> pd.DataFrame:
        """특정 모델의 평가 결과를 조회 (README 예시 구현)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
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

        except sqlite3.Error as e:
            logger.error(f"❌ 평가 결과 조회 실패: {e}")
            return pd.DataFrame()

    def get_task_leaderboard(self, task_type: str, metric_name: str,
                             dataset_name: Optional[str] = None, limit: int = 50) -> pd.DataFrame:
        """태스크별 리더보드를 생성 (README 리더보드 구현)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
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

        except sqlite3.Error as e:
            logger.error(f"❌ 리더보드 조회 실패: {e}")
            return pd.DataFrame()

    def get_model_statistics(self) -> Dict[str, Any]:
        """모델 통계 정보 반환 (README 통계 구현)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
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

        except sqlite3.Error as e:
            logger.error(f"❌ 통계 조회 실패: {e}")
            return {}

    def search_models(self, query: str, limit: int = 50) -> pd.DataFrame:
        """모델 검색 (README 검색 기능 구현)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
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

        except sqlite3.Error as e:
            logger.error(f"❌ 모델 검색 실패: {e}")
            return pd.DataFrame()

    def get_top_models_by_downloads(self, limit: int = 50, task: Optional[str] = None) -> pd.DataFrame:
        """다운로드 수 기준 상위 모델 (README 상위 모델 표 구현)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
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

        except sqlite3.Error as e:
            logger.error(f"❌ 상위 모델 조회 실패: {e}")
            return pd.DataFrame()

    def export_to_csv(self, output_dir: str = "exports"):
        """데이터를 CSV로 내보내기 (README 내보내기 구현)"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            exported_files = []

            with sqlite3.connect(self.db_path) as conn:
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

        except Exception as e:
            logger.error(f"❌ 데이터 내보내기 실패: {e}")

    def backup_database(self, backup_path: Optional[str] = None):
        """데이터베이스 백업 (README 문제 해결 구현)"""
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"data/backup_llm_evaluations_{timestamp}.db"

        try:
            import shutil
            backup_path_obj = Path(backup_path)
            backup_path_obj.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(self.db_path, backup_path)
            logger.info(f"💾 데이터베이스 백업 완료: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"❌ 데이터베이스 백업 실패: {e}")
            return None

    def vacuum_database(self):
        """데이터베이스 최적화 (README 성능 최적화 구현)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                logger.info("🔧 데이터베이스 최적화 시작...")
                conn.execute("VACUUM")
                conn.execute("ANALYZE")
                logger.info("✅ 데이터베이스 최적화 완료")
        except sqlite3.Error as e:
            logger.error(f"❌ 데이터베이스 최적화 실패: {e}")

    def get_database_size(self) -> Dict[str, Any]:
        """데이터베이스 크기 정보 조회"""
        try:
            file_size = self.db_path.stat().st_size
            size_mb = file_size / (1024 * 1024)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 테이블별 레코드 수
                tables = ['models', 'evaluations', 'task_categories', 'collection_stats']
                table_counts = {}

                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    table_counts[table] = cursor.fetchone()[0]

            return {
                'file_size_bytes': file_size,
                'file_size_mb': round(size_mb, 2),
                'table_counts': table_counts,
                'db_path': str(self.db_path)
            }

        except Exception as e:
            logger.error(f"❌ 데이터베이스 크기 조회 실패: {e}")
            return {}

    def close(self):
        """데이터베이스 연결 종료 (README 리소스 정리 구현)"""
        logger.debug("🔒 데이터베이스 관리자 종료")

    def __enter__(self):
        """컨텍스트 매니저 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.close()