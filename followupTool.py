from typing import List, Dict, Any

class QueryState:
    def __init__(self, original_question: str, required_params: List[str]):
        self.original_question = original_question
        self.required_params = required_params
        self.answered_params: Dict[str, Any] = {}
        self.pending_params = list(required_params)

    def update(self, param: str, value: Any):
        if param in self.pending_params:
            self.answered_params[param] = value
            self.pending_params.remove(param)


class QueryStateManager:
    def __init__(self):
        self.states: Dict[str, QueryState] = {}  # conv_id -> QueryState

    def create_state(self, conv_id: str, original_question: str, required_params: List[str]):
        self.states[conv_id] = QueryState(original_question, required_params)

    def update_state(self, conv_id: str, param: str, value: Any):
        if conv_id in self.states:
            self.states[conv_id].update(param, value)

    def needs_followup(self, conv_id: str) -> bool:
        return conv_id in self.states and len(self.states[conv_id].pending_params) > 0

    def get_pending(self, conv_id: str) -> List[str]:
        return self.states[conv_id].pending_params if conv_id in self.states else []

    def get_answered(self, conv_id: str) -> Dict[str, Any]:
        return self.states[conv_id].answered_params if conv_id in self.states else {}

    def clear_state(self, conv_id: str):
        if conv_id in self.states:
            del self.states[conv_id]
