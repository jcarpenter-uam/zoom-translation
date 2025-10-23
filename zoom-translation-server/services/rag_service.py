import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.schema import CreateSchema

from .debug_service import log_pipeline_step


def dbconnect():
    """
    Connects to PostgreSQL using .env variables, creates a schema,
    warns if the 'vector' extension is missing, and returns an
    engine configured to use that schema.
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
        # Create the connection URL
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
                "DATABASE", "Checking for 'vector' extension...", detailed=True
            )
            check_sql = text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            result = conn.execute(check_sql).fetchone()

            if result:
                log_pipeline_step(
                    "DATABASE",
                    "'vector' extension is correctly installed.",
                    detailed=True,
                )
            else:
                log_pipeline_step(
                    "DATABASE",
                    "WARNING: The 'vector' (pgvector) extension is NOT installed.",
                    detailed=False,
                )
                log_pipeline_step(
                    "DATABASE",
                    "The application will start, but any vector operations WILL FAIL.",
                    detailed=False,
                )
                log_pipeline_step(
                    "DATABASE",
                    "Ask an admin to connect and run: CREATE EXTENSION vector;",
                    detailed=False,
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


# function to store correction api calls and model responses in db with embeddings using ollama embedding model
# use llamaindex to determine if 2 extries are to similar to store both
