import inspect
import sys
from types import FrameType
from typing import Any, Callable, Dict, List, Optional, Set, TextIO

from sha256 import generate_hash
from Tracer import Tracer
from utils import clear_next_inputs, input, next_inputs


class CondBreakDebugger(Tracer):
    """Interactive Debugger"""

    def __init__(self, *, file: TextIO = sys.stdout) -> None:
        """Create a new interactive debugger."""
        self.stepping: bool = True
        self.breakpoints: Set[int] = set()
        self.break_conditions: Set[str] = set()
        self.interact: bool = True

        self.frame: FrameType
        self.event: Optional[str] = None
        self.arg: Any = None

        self.local_vars: Dict[str, Any] = {}

        super().__init__(file=file)

    def traceit(self, frame: FrameType, event: str, arg: Any) -> None:
        """Tracing function; called at every line. To be overloaded in subclasses."""
        self.frame = frame
        self.local_vars = frame.f_locals  # Dereference exactly once
        self.event = event
        self.arg = arg

        if self.stop_here():
            self.interaction_loop()

    def stop_here(self) -> bool:
        """Return True if we should stop"""
        for condition in self.break_conditions:
            try:
                if eval(condition, self.frame.f_globals, self.local_vars):
                    return True
            except Exception:
                continue
        return self.stepping or self.frame.f_lineno in self.breakpoints

    def interaction_loop(self) -> None:
        """Interact with the user"""
        self.print_debugger_status(self.frame, self.event, self.arg)  # type: ignore

        self.interact = True
        while self.interact:
            command = input("(debugger) ")
            self.execute(command)  # type: ignore

    def execute(self, command: str) -> None:
        """Execute `command`"""

        sep = command.find(" ")
        if sep > 0:
            cmd = command[:sep].strip()
            arg = command[sep + 1 :].strip()
        else:
            cmd = command.strip()
            arg = ""

        method = self.command_method(cmd)
        if method:
            method(arg)


    def command_method(self, command: str) -> Optional[Callable[[str], None]]:
        """Convert `command` into the method to be called.
        If the method is not found, return `None` instead."""

        if command.startswith("#"):
            return None  # Comment

        possible_cmds = [
            possible_cmd
            for possible_cmd in self.commands()
            if possible_cmd.startswith(command)
        ]
        if len(possible_cmds) != 1:
            self.help_command(command)
            return None

        cmd = possible_cmds[0]
        return getattr(self, cmd + "_command")

    def commands(self) -> List[str]:
        """Return a list of commands"""

        cmds = [
            method.replace("_command", "")
            for method in dir(self.__class__)
            if method.endswith("_command")
        ]
        cmds.sort()
        return cmds
    
    def step_command(self, arg: str = "") -> None:
        """Execute up to the next line"""

        self.stepping = True
        self.interact = False

    def continue_command(self, arg: str = "") -> None:
        """Resume execution"""

        self.stepping = False
        self.interact = False

    def help_command(self, command: str = "") -> None:
        """Give help on given `command`. If no command is given, give help on all"""

        if command:
            possible_cmds = [
                possible_cmd
                for possible_cmd in self.commands()
                if possible_cmd.startswith(command)
            ]

            if len(possible_cmds) == 0:
                self.log(f"Unknown command {repr(command)}. Possible commands are:")
                possible_cmds = self.commands()
            elif len(possible_cmds) > 1:
                self.log(f"Ambiguous command {repr(command)}. Possible expansions are:")
        else:
            possible_cmds = self.commands()

        for cmd in possible_cmds:
            method = self.command_method(cmd)
            self.log(f"{cmd:10} -- {method.__doc__}")

    def print_command(self, arg: str = "") -> None:
        """Print an expression. If no expression is given, print all variables"""

        vars = self.local_vars

        if not arg:
            self.log(
                "\n".join([f"{var} = {repr(value)}" for var, value in vars.items()])
            )
        else:
            try:
                self.log(f"{arg} = {repr(eval(arg, globals(), vars))}")
            except Exception as err:
                self.log(f"{err.__class__.__name__}: {err}")

    def list_command(self, arg: str = "") -> None:
        """Show current function. If `arg` is given, show its source code."""

        try:
            if arg:
                obj = eval(arg)
                source_lines, line_number = inspect.getsourcelines(obj)
                current_line = -1
            else:
                source_lines, line_number = inspect.getsourcelines(self.frame.f_code)
                current_line = self.frame.f_lineno
        except Exception as err:
            self.log(f"{err.__class__.__name__}: {err}")
            source_lines = []
            line_number = 0

        for line in source_lines:
            spacer = " "
            if line_number == current_line:
                spacer = ">"
            elif line_number in self.breakpoints:
                spacer = "#"
            self.log(f"{line_number:4}{spacer} {line}", end="")
            line_number += 1

    def break_command(self, arg: str = "") -> None:
        """Set a breakoint in given line. If no line is given, list all breakpoints"""
        if arg:
            try:
                break_line = int(arg)
                self.breakpoints.add(break_line)
            except ValueError:
                self.break_conditions.add(arg)
            
        self.log("Breakpoints (lines):", self.breakpoints)
        self.log("Breakpoints (conditions):", self.break_conditions)

    def delete_command(self, arg: str = "") -> None:
        """Delete breakoint in line given by `arg`.
        Without given line, clear all breakpoints"""

        if arg:
            try:
                self.breakpoints.remove(int(arg))
            except KeyError:
                self.log(f"No such breakpoint: {arg}")
            except ValueError:
                try:
                    self.break_conditions.remove(arg)
                except ValueError:
                    self.log(f"No such breakpoint: {arg}")
        else:
            self.breakpoints = set()
        self.log("Breakpoints (lines):", self.breakpoints)
        self.log("Breakpoints (conditions):", self.break_conditions)

    def quit_command(self, arg: str = "") -> None:
        """Finish execution"""

        self.breakpoints = set()
        self.break_conditions = set()
        self.stepping = False
        self.interact = False

    def assign_command(self, arg: str) -> None:
        """Use as 'assign VAR=VALUE'. Assign VALUE to local variable VAR."""

        sep = arg.find("=")
        if sep > 0:
            var = arg[:sep].strip()
            expr = arg[sep + 1 :].strip()
        else:
            self.help_command("assign")
            return

        vars = self.local_vars
        try:
            vars[var] = eval(expr, self.frame.f_globals, vars)
        except Exception as err:
            self.log(f"{err.__class__.__name__}: {err}")

    def attr_command(self, arg: str) -> None:
        if arg:
            obj, var, expr = arg.split(",")
            obj = obj.strip()
            var = var.strip()
            expr = expr.strip()
        
            vars = self.local_vars
            result = eval(expr, self.frame.f_globals, vars)
            setattr(vars[obj], var, result)

    def set_command(self, arg: str) -> None:
        """Use as 'set VAR=VALUE'. Set VALUE to local variable VAR."""

        sep = arg.find("=")
        if sep > 0:
            var = arg[:sep].strip()
            expr = arg[sep + 1 :].strip()
        else:
            self.help_command("set")
            return

        vars = self.local_vars
        try:
            vars[var] = eval(expr, self.frame.f_globals, vars)
        except Exception as err:
            self.log(f"{err.__class__.__name__}: {err}")

def foo():
    a = 10
    b = 5
    a = 15
    a = 5

if __name__ == "__main__":
    with CondBreakDebugger():
        foo()