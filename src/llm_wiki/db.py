import os
import pymysql
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OB_HOST = os.getenv("OB_HOST", "127.0.0.1")
OB_PORT = int(os.getenv("OB_PORT", 2881))
OB_USER = os.getenv("OB_USER", "root")
OB_PASSWORD = os.getenv("OB_PASSWORD", "")
OB_DATABASE = os.getenv("OB_DATABASE", "llm_wiki")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1536))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

def get_db_connection():
    return pymysql.connect(
        host=OB_HOST,
        port=OB_PORT,
        user=OB_USER,
        password=OB_PASSWORD,
        database=OB_DATABASE,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True
    )

def get_config(conn, key):
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT config_value FROM wiki_config WHERE config_key = %s", (key,))
            result = cursor.fetchone()
            if result:
                return result['config_value']
        return None
    except pymysql.err.ProgrammingError:
        # Table does not exist yet — treated as first-time setup
        return None
    except Exception as e:
        raise RuntimeError(f"Failed to read config key '{key}' from database") from e

def set_config(conn, key, value):
    with conn.cursor() as cursor:
        cursor.execute("""
            INSERT INTO wiki_config (config_key, config_value)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE config_value = %s
        """, (key, value, value))

def initialize_base_schema(conn):
    """Initialize the base tracking tables that don't depend on vector dimension."""
    with conn.cursor() as cursor:
        # Create config table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS wiki_config (
                config_key VARCHAR(64) PRIMARY KEY,
                config_value VARCHAR(255) NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        """)

        # Create documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS wiki_documents (
                doc_id VARCHAR(64) PRIMARY KEY,
                file_path VARCHAR(512) NOT NULL,
                category VARCHAR(64) NOT NULL,
                file_hash VARCHAR(64) NOT NULL,
                embedding_model VARCHAR(128) NOT NULL,
                last_sync_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uk_file_path (file_path)
            )
        """)

def ensure_schema_and_model_match(conn, current_model, current_dim, current_chunk_size, current_chunk_overlap):
    """
    Checks if the active model and chunking parameters match the DB.
    If not, it wipes the vector table and recreates it with the correct dimension.
    """
    initialize_base_schema(conn)

    db_model = get_config(conn, 'ACTIVE_EMBEDDING_MODEL')
    db_chunk_size = get_config(conn, 'ACTIVE_CHUNK_SIZE')
    db_chunk_overlap = get_config(conn, 'ACTIVE_CHUNK_OVERLAP')

    model_changed = (db_model != current_model)
    chunk_changed = (db_chunk_size != str(current_chunk_size) or db_chunk_overlap != str(current_chunk_overlap))

    if model_changed or chunk_changed:
        if db_model is None:
            print(f"[*] First time setup: Initializing with model {current_model} (dim={current_dim}), chunk_size={current_chunk_size}")
        else:
            reasons = []
            if model_changed:
                reasons.append(f"Model ({db_model} -> {current_model})")
            if chunk_changed:
                reasons.append(f"Chunk parameters changed")
            print(f"[!] Change detected: {', '.join(reasons)}. Rebuilding vector tables...")

        with conn.cursor() as cursor:
            # 1. Drop existing chunk tables and clear documents mapping
            cursor.execute("DROP TABLE IF EXISTS wiki_chunks")
            # We clear documents because the chunks are gone; they need full re-sync
            cursor.execute("TRUNCATE TABLE wiki_documents")

            # 2. Create new chunks table with dynamic vector dimension
            create_chunks_sql = f"""
                CREATE TABLE wiki_chunks (
                    chunk_id VARCHAR(64) PRIMARY KEY,
                    doc_id VARCHAR(64) NOT NULL,
                    file_path VARCHAR(512) NOT NULL,
                    chunk_index INT NOT NULL,
                    content TEXT NOT NULL,
                    embedding VECTOR({current_dim}) NOT NULL,
                    CONSTRAINT fk_doc FOREIGN KEY (doc_id) REFERENCES wiki_documents(doc_id) ON DELETE CASCADE
                )
            """
            cursor.execute(create_chunks_sql)

            # 3. Create HNSW Vector Index
            try:
                print("[*] Creating HNSW Vector Index...")
                cursor.execute("""
                    CREATE VECTOR INDEX idx_wiki_chunks_embedding
                    ON wiki_chunks (embedding)
                    WITH (DISTANCE_MEASURE = 'COSINE', TYPE = 'HNSW')
                """)
            except Exception as e:
                print(f"[!] Warning: Could not create vector index (requires OceanBase 4.3.3+): {e}")

        # Update config
        set_config(conn, 'ACTIVE_EMBEDDING_MODEL', current_model)
        set_config(conn, 'ACTIVE_EMBEDDING_DIM', str(current_dim))
        set_config(conn, 'ACTIVE_CHUNK_SIZE', str(current_chunk_size))
        set_config(conn, 'ACTIVE_CHUNK_OVERLAP', str(current_chunk_overlap))
        print("[*] Schema matches current embedding model and chunking parameters.")
    else:
        print(f"[*] Schema is up-to-date for model {current_model} and chunk parameters.")

if __name__ == "__main__":
    print("Connecting to OceanBase...")
    try:
        connection = get_db_connection()
        ensure_schema_and_model_match(connection, EMBEDDING_MODEL, EMBEDDING_DIM, CHUNK_SIZE, CHUNK_OVERLAP)
        connection.close()
    except Exception as e:
        print(f"[X] Error connecting to database or initializing: {e}")