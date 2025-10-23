import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import ollama
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.schema import CreateSchema

from .debug_service import log_pipeline_step


def dbconnect():
    """
    Connects to PostgreSQL using .env variables, creates a schema,
    and returns an engine configured to use that schema.
    """
    load_dotenv()

    db_name = os.getenv("POSTGRES_DB_NAME")
    db_user = os.getenv("POSTGRES_DB_USER")
    db_password = os.getenv("POSTGRES_DB_PASSWORD")
    db_host = os.getenv("POSTGRES_DB_HOST")
    db_port = os.getenv("POSTGRES_DB_PORT")
    schema_name = os.getenv("POSTGRES_DB_SCHEMA")

    if not all([db_name, db_user, db_password, db_host, db_port, schema_name]):
        log_pipeline_step(
            "DATABASE",
            "Error: Not all required database environment variables are set.",
            detailed=False,
        )
        log_pipeline_step(
            "DATABASE",
            "Check: POSTGRES_DB_NAME, POSTGRES_DB_USER, POSTGRES_DB_PASSWORD, POSTGRES_DB_HOST, POSTGRES_DB_PORT, POSTGRES_DB_SCHEMA",
            detailed=False,
        )
        return None

    try:
        connection_url = URL.create(
            drivername="postgresql+psycopg2",
            username=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
            database=db_name,
        )

        setup_engine = create_engine(connection_url, echo=False)

        with setup_engine.begin() as conn:
            log_pipeline_step(
                "DATABASE",
                f"Successfully connected to database '{db_name}' at {db_host}.",
                detailed=True,
            )

            log_pipeline_step(
                "DATABASE",
                f"Checking/creating schema: '{schema_name}'...",
                detailed=True,
            )
            conn.execute(CreateSchema(schema_name, if_not_exists=True))
            log_pipeline_step(
                "DATABASE", f"Schema '{schema_name}' is ready.", detailed=True
            )

        log_pipeline_step(
            "DATABASE", "Database setup commands complete.", detailed=False
        )

        app_engine = create_engine(
            connection_url,
            connect_args={"options": f"-csearch_path={schema_name}"},
            echo=False,
        )

        with app_engine.connect() as conn:
            result = conn.execute(text("SHOW search_path"))
            log_pipeline_step(
                "DATABASE",
                f"Engine configured. Default search_path: {result.fetchone()[0]}",
                detailed=True,
            )

        return app_engine

    except OperationalError as e:
        log_pipeline_step(
            "DATABASE",
            "Error: Database connection failed. Check .env and if PostgreSQL is running.",
            detailed=False,
        )
        log_pipeline_step("DATABASE", f"Details: {e}", detailed=True)
        return None
    except ProgrammingError as e:
        log_pipeline_step(
            "DATABASE",
            "Error: A database command failed. Check user permissions.",
            detailed=False,
        )
        log_pipeline_step("DATABASE", f"Details: {e}", detailed=True)
        return None
    except Exception as e:
        log_pipeline_step(
            "DATABASE",
            f"An unexpected error occurred during database setup: {e}",
            detailed=False,
        )
        return None


class RagService:
    """
    Manages database operations for logging corrections and generating embeddings.
    """

    def __init__(self, ollama_url: str, engine: Engine, model: str = "embeddinggemma"):
        """
        Initializes the RAG service.

        Args:
            ollama_url (str): The base URL for the Ollama API.
            engine (Engine): The SQLAlchemy engine for database connections.
            model (str): The name of the embedding model to use.
        """
        log_pipeline_step(
            "RAG",
            f"Initializing RAG Service with model '{model}' at {ollama_url}...",
            detailed=False,
        )
        self.model = model
        self.engine = engine
        self.client = ollama.AsyncClient(host=ollama_url)
        self.is_vector_installed = self._check_extensions()

        log_pipeline_step(
            "RAG",
            "Ollama embedding client initialized.",
            extra={
                "model": model,
                "host": ollama_url,
                "pgvector_enabled": self.is_vector_installed,
            },
            detailed=True,
        )

        self.create_rag_tables()

    # BUG: if the uuid-ossp extension is not installed the schema never gets created and nothing gets saved, implement the failure logic
    def _check_extensions(self) -> bool:
        """
        Checks if the extensions are installed in the database.
        """
        log_pipeline_step("DATABASE", "Checking for extensions...", detailed=True)
        try:
            with self.engine.begin() as conn:
                check_sql = text(
                    "SELECT COUNT(*) FROM pg_extension WHERE extname IN ('vector', 'uuid-ossp')"
                )
                result = conn.execute(check_sql).fetchone()

                if result:
                    log_pipeline_step(
                        "DATABASE",
                        "vector & uuid-ossp extensions are correctly installed. Embedding column will be used.",
                        detailed=True,
                    )
                    return True
                else:
                    log_pipeline_step(
                        "DATABASE",
                        "WARNING: The vector & uuid-ossp extensions are NOT installed.",
                        detailed=False,
                    )
                    return False
        except Exception as e:
            log_pipeline_step(
                "DATABASE",
                f"Error checking for 'vector' extension: {e}. Assuming it is not installed.",
                detailed=False,
            )
            return False

    def create_rag_tables(self):
        """
        Creates the 'correction_logs' table in the database.
        The 'embedding' column is only added if the 'vector' extension is present.
        """
        log_pipeline_step("RAG", "Attempting to create 'correction_logs' table...")

        embedding_column_sql = ""
        if self.is_vector_installed:
            embedding_column_sql = "embedding vector(768),"

        create_table_sql = text(
            f"""
            CREATE TABLE IF NOT EXISTS correction_logs (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                original_transcription TEXT NOT NULL,
                original_translation TEXT NOT NULL,
                context_history JSONB,
                corrected_transcription TEXT NOT NULL,
                corrected_translation TEXT NOT NULL,
                correction_reason TEXT,
                correction_confidence FLOAT,
                {embedding_column_sql}
                metadata JSONB
            );
        """
        )

        try:
            with self.engine.begin() as conn:
                conn.execute(create_table_sql)

            log_pipeline_step(
                "RAG", "Table 'correction_logs' checked/created successfully."
            )

        except Exception as e:
            log_pipeline_step(
                "RAG",
                f"Error creating 'correction_logs' table: {e}. This may be critical.",
            )

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generates an embedding for a given text using the service's model.
        Returns None on failure.
        """
        try:
            response = await self.client.embeddings(model=self.model, prompt=text)
            log_pipeline_step(
                "RAG", f"Generated embedding using '{self.model}'.", detailed=True
            )
            return response.get("embedding")
        except Exception as e:
            log_pipeline_step(
                "RAG",
                f"Error generating embedding with model '{self.model}': {e}. Proceeding without embedding.",
            )
            return None

    async def log_correction(
        self,
        original_transcription: str,
        original_translation: str,
        context_history: List[str],
        corrected_transcription: str,
        corrected_translation: str,
        correction_reason: str,
        correction_confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Stores a correction event in the database.
        If pgvector is enabled, it also generates and stores an embedding.
        If embedding fails, it stores NULL but saves the rest of the data.
        """
        log_pipeline_step("RAG", "Attempting to log correction to DB...")

        params = {
            "original_transcription": original_transcription,
            "original_translation": original_translation,
            "context_history": json.dumps(context_history),
            "corrected_transcription": corrected_transcription,
            "corrected_translation": corrected_translation,
            "correction_reason": correction_reason,
            "correction_confidence": correction_confidence,
            "metadata": json.dumps(metadata) if metadata else None,
        }

        table_name = "correction_logs"
        column_names = [
            "original_transcription",
            "original_translation",
            "context_history",
            "corrected_transcription",
            "corrected_translation",
            "correction_reason",
            "correction_confidence",
            "metadata",
        ]
        column_values = [
            ":original_transcription",
            ":original_translation",
            ":context_history",
            ":corrected_transcription",
            ":corrected_translation",
            ":correction_reason",
            ":correction_confidence",
            ":metadata",
        ]

        if self.is_vector_installed:
            embedding = await self.get_embedding(original_transcription)
            column_names.append("embedding")
            column_values.append(":embedding")
            params["embedding"] = str(embedding) if embedding else None

        columns_sql = ", ".join(column_names)
        values_sql = ", ".join(column_values)

        insert_sql = text(
            f"""
            INSERT INTO {table_name} ({columns_sql})
            VALUES ({values_sql})
            """
        )

        def _db_insert():
            try:
                with self.engine.begin() as conn:
                    conn.execute(insert_sql, params)
                return True
            except Exception as e:
                log_pipeline_step("RAG", f"Database insert failed: {e}")
                return False

        try:
            success = await asyncio.to_thread(_db_insert)
            if success:
                log_pipeline_step("RAG", "Successfully logged correction to database.")

        except Exception as e:
            log_pipeline_step(
                "RAG", f"An unexpected error occurred during logging: {e}"
            )


# TODO: use llamaindex to determine if 2 extries are to similar to store both
