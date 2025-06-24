import datetime
import os
import json
import time

class SessionManager:
    def __init__(self):
        now = datetime.datetime.now()
        self.session_id = now.strftime("session_%Y%m%d_%H%M%S")
        self.timestamp = now.isoformat()

        self.query_log = []
        self.current_results = []
        self.last_query = None
        self.original_query = None

        self.refinements = []
        self.undo_stack = []  # ← NEW: Tracks last state for undo

    def update(self, query, officers, is_refinement=False):
        self.query_log.append(query)

        # Always push current state into undo stack BEFORE updating it
        if self.current_results:
            self.undo_stack.append({
                "query": self.last_query,
                "result": self.current_results
            })

        self.current_results = officers
        self.last_query = query

        if not is_refinement:
            self.original_query = query
        else:
            self.refinements.append({
                "query": query,
                "result": officers
            })

    def reset(self):
        self.query_log = []
        self.current_results = []
        self.last_query = None
        self.original_query = None
        self.refinements = []
        self.undo_stack = []

    def get_current_pool(self):
        return self.current_results

    def get_session_data(self, final_result):
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "original_query": self.original_query,
            "refinements": self.refinements,
            "final_result": final_result
        }

    def save_session(self, final_result, folder="sessions"):
        os.makedirs(folder, exist_ok=True)
        data = self.get_session_data(final_result)
        save_path = os.path.join(folder, f"{self.session_id}.json")
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Session saved to: {save_path}")

    def undo_last(self):
        if not self.undo_stack:
            print("⚠️ Nothing to undo.")
            return None

        # Pop the last state
        last = self.undo_stack.pop()
        self.current_results = last["result"]
        self.last_query = last["query"]

        # Return it in JSON format
        response = {
            "results": self.current_results,
            "confidence": "restored"
        }
        return json.dumps(response, indent=2)

    def log_score(self, query, score, source, was_fallback=False, force_override=False):
        """Log metadata related to query score and fallback usage."""
        log_entry = {
            "query": query,
            "score": score,
            "source": source,
            "was_fallback": was_fallback,
            "forced_override": force_override,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        if not hasattr(self, "score_log"):
            self.score_log = []
        self.score_log.append(log_entry)
        print("📊 Score log entry added:", log_entry)