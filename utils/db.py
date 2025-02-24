import sqlite3
import os
import shutil

def clear_sqlite() -> str:
    sqlite_folder = ".db//chroma_langchain_db//chroma.sqlite3"
    conn = sqlite3.connect(sqlite_folder)
    cursor = conn.cursor()
    cursor.execute("VACUUM;")
    conn.commit()
    conn.close()
    return "Database cleared!"

def clear_folder():
    db_folder = ".db//chroma_langchain_db"
    if os.path.exists(db_folder):
        for filename in os.listdir(db_folder):
            file_path = os.path.join(db_folder, filename)
            try:
                if filename == "chroma.sqlite3":
                    continue
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error: {e}")

