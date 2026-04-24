
import hashlib
import uuid
from chonkie import SentenceChunker
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv




from llm_wiki.db import get_db_connection, ensure_schema_and_model_match
import os
import fitz

# Load env
load_dotenv()

# Constants
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1536))
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

# Initialize OpenAI Client
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE if OPENAI_API_BASE else None
)

def calculate_file_hash(filepath):
    """Calculate MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def get_existing_documents(conn):
    """Return a dictionary of file_path -> file_hash for existing docs."""
    with conn.cursor() as cursor:
        cursor.execute("SELECT file_path, file_hash FROM wiki_documents")
        return {row['file_path']: row['file_hash'] for row in cursor.fetchall()}

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chunk text using chonkie SentenceChunker."""
    chunker = SentenceChunker(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunks = chunker(text)
    return [chunk.text for chunk in chunks]

def get_embeddings(texts):
    """Fetch embeddings for a list of texts from OpenAI/Compatible API."""
    # Process in batches to avoid API limits
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            input=batch,
            model=EMBEDDING_MODEL
        )
        batch_embeddings = [data.embedding for data in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings

def extract_text_from_file(filepath: str) -> str:
    """Extracts text depending on the file type."""
    ext = filepath.lower().split('.')[-1]

    if ext == 'md':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    elif ext == 'pdf':
        content = []
        # Open the PDF using PyMuPDF
        with fitz.open(filepath) as doc:
            for page in doc:
                content.append(page.get_text())

        # Join extracted text
        return "\n".join(content)

    return ""

def sync_file(conn, filepath):
    """Process a single file: chunk, embed, and save to DB."""
    file_hash = calculate_file_hash(filepath)
    # Extract directory name as category (e.g., 'wiki/concepts' -> 'concepts')
    dir_name = os.path.basename(os.path.dirname(filepath))
    category = dir_name if dir_name else 'root'

    # Read content
    content = extract_text_from_file(filepath)

    if not content.strip():
        return # Skip empty files

    chunks = chunk_text(content)
    if not chunks:
        return

    print(f"  Generating embeddings for {len(chunks)} chunks in {filepath}...")
    embeddings = get_embeddings(chunks)

    doc_id = str(uuid.uuid4())

    # DB Transaction
    with conn.cursor() as cursor:
        # 1. Delete old chunks/doc if file existed before
        cursor.execute("DELETE FROM wiki_documents WHERE file_path = %s", (filepath,))

        # 2. Insert document record
        cursor.execute("""
            INSERT INTO wiki_documents (doc_id, file_path, category, file_hash, embedding_model)
            VALUES (%s, %s, %s, %s, %s)
        """, (doc_id, filepath, category, file_hash, EMBEDDING_MODEL))

        # 3. Prepare chunks
        chunk_data = []
        for idx, (chunk_text_content, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_{idx}"
            # OceanBase expects a string format like '[0.1,0.2,...]' for vector insertion
            # We must make sure it converts correctly from a float list to a string
            emb_str = str(list(emb))
            chunk_data.append((chunk_id, doc_id, filepath, idx, chunk_text_content, emb_str))

        # 4. Insert chunks
        cursor.executemany("""
            INSERT INTO wiki_chunks (chunk_id, doc_id, file_path, chunk_index, content, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, chunk_data)

def run_sync():
    conn = get_db_connection()

    # 1. Ensure Schema matches current model and chunk parameters
    print("Checking database schema and model configuration...")
    ensure_schema_and_model_match(conn, EMBEDDING_MODEL, EMBEDDING_DIM, CHUNK_SIZE, CHUNK_OVERLAP)

    # 2. Get existing state
    existing_docs = get_existing_documents(conn)

    # 3. Scan files
    directories_to_scan = ['raw', 'wiki']
    files_to_sync = []

    for directory in directories_to_scan:
        if not os.path.exists(directory):
            continue
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.md', '.pdf')):
                    # Use forward slashes for cross-platform consistency
                    filepath = os.path.join(root, file).replace('\\', '/')
                    files_to_sync.append(filepath)

    # 4. Compare and Sync
    processed_count = 0
    for filepath in tqdm(files_to_sync, desc="Scanning files"):
        current_hash = calculate_file_hash(filepath)

        # If file is new or hash changed
        if filepath not in existing_docs or existing_docs[filepath] != current_hash:
            print(f"\nSyncing changed/new file: {filepath}")
            try:
                sync_file(conn, filepath)
                processed_count += 1
            except Exception as e:
                print(f"[X] Failed to sync {filepath}: {e}")

    # 5. Cleanup deleted files
    current_files_set = set(files_to_sync)
    db_files_set = set(existing_docs.keys())
    deleted_files = db_files_set - current_files_set

    if deleted_files:
        with conn.cursor() as cursor:
            for df in deleted_files:
                print(f"Removing deleted file from index: {df}")
                cursor.execute("DELETE FROM wiki_documents WHERE file_path = %s", (df,))

    conn.close()
    print(f"\n[*] Sync complete. Processed {processed_count} files. Removed {len(deleted_files)} files.")

if __name__ == "__main__":
    run_sync()