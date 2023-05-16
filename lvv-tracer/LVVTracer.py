from typing import Optional, Type, Any, Callable
from types import TracebackType, FrameType
from collections import defaultdict
import sys
from copy import deepcopy


class LVVTracer:
    def __init__(self, target_func: str):
        self.original_trace_function: Optional[Callable] = None
        self.target = target_func
        self.locals = dict()
        self.change_count = defaultdict(int)

    def __enter__(self) -> Any:
        self.original_trace_function = sys.gettrace()
        sys.settrace(self._traceit)
        return self

    def traceit(self, frame: FrameType, event: str, arg: Any) -> None:
        if event == "call":
            return None
        for key, value in frame.f_locals.items():
            if key in self.locals and self.locals[key] == value:
                pass
            else:
                self.change_count[key] += 1
        self.locals.update(deepcopy(frame.f_locals))

    def _traceit(self, frame: FrameType, event: str, arg: Any) -> Optional[Callable]:
        if event == "call" and frame.f_code.co_name == self.target:
            self.locals.clear()
            for key, value in frame.f_locals.items():
                if key in self.locals and self.locals[key] == value:
                    pass
                else:
                    self.change_count[key] += 1
            self.locals.update(deepcopy(frame.f_locals))
            return self.traceit

    def __exit__(
        self, exc_tp: Type, exc_value: BaseException, exc_traceback: TracebackType
    ) -> Optional[bool]:
        sys.settrace(self.original_trace_function)
        return False

    def getLVVmap(self) -> dict:
        return self.change_count
