"""Local SQLite asset library for generated Bunyaan assets."""
import sqlite3, json, uuid
from pathlib import Path
from datetime import datetime, timezone

class AssetLibrary:
    def __init__(self, db_path="assets/bunyaan-library.sqlite3"):
        self.path=Path(db_path); self.path.parent.mkdir(parents=True,exist_ok=True)
        with self._db() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS assets (
              id TEXT PRIMARY KEY, name TEXT NOT NULL, prompt TEXT NOT NULL,
              file_path TEXT NOT NULL, format TEXT NOT NULL, created_at TEXT NOT NULL,
              metadata TEXT NOT NULL DEFAULT '{}')''')
    def _db(self):
        db=sqlite3.connect(self.path); db.row_factory=sqlite3.Row; return db
    def add(self,name,prompt,file_path,metadata=None):
        aid=str(uuid.uuid4()); path=str(Path(file_path).resolve())
        with self._db() as db:
            db.execute('INSERT INTO assets VALUES (?,?,?,?,?,?,?)',(aid,name,prompt,path,Path(path).suffix.lower().lstrip('.'),datetime.now(timezone.utc).isoformat(),json.dumps(metadata or {})))
        return self.get(aid)
    def get(self,asset_id):
        with self._db() as db: row=db.execute('SELECT * FROM assets WHERE id=?',(asset_id,)).fetchone()
        return dict(row) if row else None
    def list(self,limit=100,query=None):
        with self._db() as db:
            if query:
                rows=db.execute('SELECT * FROM assets WHERE name LIKE ? OR prompt LIKE ? ORDER BY created_at DESC LIMIT ?', (f'%{query}%',f'%{query}%',limit)).fetchall()
            else: rows=db.execute('SELECT * FROM assets ORDER BY created_at DESC LIMIT ?',(limit,)).fetchall()
        return [dict(x) for x in rows]
    def delete(self,asset_id,delete_file=False):
        item=self.get(asset_id)
        if not item:return False
        with self._db() as db: db.execute('DELETE FROM assets WHERE id=?',(asset_id,))
        if delete_file:
            p=Path(item['file_path'])
            if p.exists(): p.unlink()
        return True
