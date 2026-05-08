import os
import sys
import argparse
from openai import OpenAI
from dotenv import load_dotenv

from llm_wiki.db import get_db_connection

# Load env
load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Client
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE if OPENAI_API_BASE else None
)

def search_wiki(query, top_k=5, path_filter=None):
    """
    Search the wiki for the given query using OceanBase Vector Search.
    Returns the top K results.
    """
    print(f"🔍 Searching for: '{query}'", file=sys.stderr)

    # 1. Generate embedding for query
    try:
        response = client.embeddings.create(
            input=query,
            model=EMBEDDING_MODEL
        )
        query_vector = response.data[0].embedding
        query_vector_str = str(query_vector)
    except Exception as e:
        print(f"[!] Failed to generate embedding: {e}", file=sys.stderr)
        return []

    # 2. Connect to DB and Execute Hybrid Search
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:

            # Parameterized vector query — avoids SQL injection (CWE-89)
            base_sql = """
                SELECT
                    file_path,
                    content,
                    COSINE_DISTANCE(embedding, %s) as distance
                FROM wiki_chunks
            """

            params = [query_vector_str]

            if path_filter:
                base_sql += " WHERE file_path LIKE %s"
                params.append(f"%{path_filter}%")

            base_sql += " ORDER BY distance ASC LIMIT %s"
            params.append(int(top_k))

            cursor.execute(base_sql, params)
            results = cursor.fetchall()

        return results

    except Exception as e:
        print(f"[X] Database search failed: {e}", file=sys.stderr)
        return []
    finally:
        if conn:
            conn.close()

def main():
    parser = argparse.ArgumentParser(description="Search LLM Wiki using OceanBase Vector Database")
    parser.add_argument("query", type=str, help="The search query")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return (default: 5)")
    parser.add_argument("--filter", type=str, default=None, help="Filter by file path (e.g., 'wiki/concepts')")

    args = parser.parse_args()

    results = search_wiki(args.query, args.k, args.filter)

    if not results:
        print("No results found.")
        return

    print("\n" + "="*80)
    print(f"TOP {len(results)} RESULTS FOR: '{args.query}'")
    print("="*80 + "\n")

    for i, r in enumerate(results, 1):
        print(f"[{i}] File: {r['file_path']} (Distance: {r['distance']:.4f})")
        print("-" * 40)
        print(r['content'])
        print("=" * 80 + "\n")

if __name__ == "__main__":
    main()