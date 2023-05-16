import sys
from types import TracebackType, FrameType
from typing import Optional, Any, Callable, Type
from copy import deepcopy
from collections import defaultdict

class LFTracer():
    def __init__(self, target_func: str = [], list_func: bool = False, file = sys.stdout) -> None:
        self.original_trace_func = None
        self.target_func = target_func
        self.list_func = isinstance(target_func, list)
        self.file = file

        self.counter = dict()

    def __enter__(self) -> Any:
        self.original_trace_function = sys.gettrace()
        sys.settrace(self._traceit)
        return self

    def should_trace(self, func) -> bool:
        if self.list_func:
            return (func in self.target_func)
        else:
            return func == self.target_func

    def traceit(self, frame: FrameType, event: str, arg: Any) -> None:
        if event == "call":
            return None
        if frame.f_lineno not in self.counter[frame.f_code.co_name]:
            self.counter[frame.f_code.co_name][frame.f_lineno] = 1
        else:
            self.counter[frame.f_code.co_name][frame.f_lineno] += 1

    def _traceit(self, frame: FrameType, event: str, arg: Any) -> Optional[Callable]:
        if event == "call" and self.should_trace(frame.f_code.co_name):
            if frame.f_code.co_name not in self.counter.keys():
                self.counter[frame.f_code.co_name] = {}
            if frame.f_lineno not in self.counter[frame.f_code.co_name]:
                self.counter[frame.f_code.co_name][frame.f_lineno] = 1
            else:
                self.counter[frame.f_code.co_name][frame.f_lineno] += 1
            return self.traceit

    def __exit__(
        self, exc_tp: Type, exc_value: BaseException, exc_traceback: TracebackType
    ) -> Optional[bool]:
        sys.settrace(self.original_trace_function)
        return False

    def getLFMap(self):
        return self.counter
