"""
SQLite-backed User Session Isolation Store for Darukaa.Earth.
Ensures strict per-user privacy:
- Each user only ever accesses their own consultation dialogues.
- No global fallbacks or cross-user data leakage.
- Automatic cleanup of stale sessions.
"""

import os
import json
import time
import sqlite3
from typing import Dict, Any, Tuple, Optional
from core.schemas import IntelligenceResponse, ChatMessage
from core.intelligence_agent import BiodiversityIntelligenceAgent


class UserSessionDB:
    """
    Manages isolated user consultations in a dedicated SQLite database.
    Each user has a unique ID, ensuring strict isolation between different visitors.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
            os.makedirs(data_dir, exist_ok=True)
            self.db_path = os.path.join(data_dir, "darukaa_sessions.db")
        else:
            self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_consultations (
                        user_id TEXT PRIMARY KEY,
                        active_session_id TEXT NOT NULL,
                        consultations_json TEXT NOT NULL,
                        updated_at REAL NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_updated 
                    ON user_consultations(updated_at)
                """)
        finally:
            conn.close()

    def save_user_consultations(self, user_id: str, consultations: Dict[str, Any], active_id: str):
        """Saves consultations strictly isolated to the given user_id."""
        if not user_id:
            return

        # Only persist if there are actual conversations or non-empty state
        has_content = any(len(s.get("chat_history", [])) > 0 for s in consultations.values())
        if not has_content:
            return

        try:
            serial_data = {}
            for s_id, s_data in consultations.items():
                chat_list = []
                for msg in s_data.get("chat_history", []):
                    item = {
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    }
                    resp = msg.get("response_obj")
                    if resp is not None:
                        if hasattr(resp, "model_dump"):
                            item["response_dict"] = resp.model_dump()
                        elif isinstance(resp, dict):
                            item["response_dict"] = resp
                    chat_list.append(item)

                agent_obj = s_data.get("agent")
                serial_data[s_id] = {
                    "title": s_data.get("title", "Consultation"),
                    "chat_history": chat_list,
                    "accumulated_params": getattr(agent_obj, "accumulated_params", {}) if agent_obj else {},
                    "gemini_api_key": getattr(agent_obj, "gemini_api_key", None) if agent_obj else None
                }

            json_payload = json.dumps(serial_data, ensure_ascii=False)
            now = time.time()

            conn = self._get_connection()
            try:
                with conn:
                    conn.execute("""
                        INSERT INTO user_consultations (user_id, active_session_id, consultations_json, updated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(user_id) DO UPDATE SET
                            active_session_id = excluded.active_session_id,
                            consultations_json = excluded.consultations_json,
                            updated_at = excluded.updated_at
                    """, (user_id, active_id, json_payload, now))
            finally:
                conn.close()
        except Exception:
            pass

    def load_user_consultations(self, user_id: str, vector_store) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Loads consultations ONLY for this user_id.
        Returns (None, None) if user has no record or user_id is unknown.
        Strictly zero fallback to other users' records.
        """
        if not user_id:
            return None, None

        try:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "SELECT active_session_id, consultations_json FROM user_consultations WHERE user_id = ?",
                    (user_id,)
                )
                row = cursor.fetchone()
            finally:
                conn.close()

            if not row:
                return None, None

            active_id = row["active_session_id"]
            saved_dict = json.loads(row["consultations_json"])

            restored = {}
            for s_id, s_data in saved_dict.items():
                restored_chat = []
                for msg in s_data.get("chat_history", []):
                    item = {
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    }
                    if "response_dict" in msg and msg["response_dict"]:
                        try:
                            item["response_obj"] = IntelligenceResponse(**msg["response_dict"])
                        except Exception:
                            item["response_obj"] = None
                    restored_chat.append(item)

                agent_inst = BiodiversityIntelligenceAgent(
                    gemini_api_key=s_data.get("gemini_api_key"),
                    vector_store=vector_store
                )
                agent_inst.accumulated_params = s_data.get("accumulated_params", {})
                for m in restored_chat:
                    agent_inst.conversation_memory.append(ChatMessage(role=m["role"], content=m["content"]))

                restored[s_id] = {
                    "title": s_data.get("title", "Consultation"),
                    "chat_history": restored_chat,
                    "agent": agent_inst
                }

            if not active_id or active_id not in restored:
                active_id = list(restored.keys())[0] if restored else None

            return restored, active_id
        except Exception:
            return None, None

    def delete_user_consultations(self, user_id: str):
        """Permanently clears history for this user_id."""
        if not user_id:
            return
        try:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute("DELETE FROM user_consultations WHERE user_id = ?", (user_id,))
            finally:
                conn.close()
        except Exception:
            pass

    def cleanup_old_sessions(self, max_age_days: int = 7):
        """Purges sessions older than max_age_days to keep database lightweight."""
        cutoff = time.time() - (max_age_days * 86400)
        try:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute("DELETE FROM user_consultations WHERE updated_at < ?", (cutoff,))
            finally:
                conn.close()
        except Exception:
            pass
