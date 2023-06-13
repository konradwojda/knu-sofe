import html
import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast

from bookutils import print_content
from graphviz import Digraph
from StackInspector import StackInspector

Location = Tuple[Callable, int]
Node = Tuple[str, Location]
Dependency = Dict[Node, Set[Node]]
Criterion = Union[str, Location, Node]

class Dependencies(StackInspector):
    """A dependency graph"""

    NODE_COLOR = 'peachpuff'
    FONT_NAME = 'Courier'  # 'Fira Mono' may produce warnings in 'dot'

    def __init__(self, 
                 data: Optional[Dependency] = None,
                 control: Optional[Dependency] = None) -> None:
        """
        Create a dependency graph from `data` and `control`.
        Both `data` and `control` are dictionaries
        holding _nodes_ as keys and sets of nodes as values.
        Each node comes as a tuple (variable_name, location)
        where `variable_name` is a string 
        and `location` is a pair (function, lineno)
        where `function` is a callable and `lineno` is a line number
        denoting a unique location in the code.
        """

        if data is None:
            data = {}
        if control is None:
            control = {}

        self.data = data
        self.control = control

        for var in self.data:
            self.control.setdefault(var, set())
        for var in self.control:
            self.data.setdefault(var, set())

        self.validate()

    def validate(self) -> None:
        """Check dependency structure."""
        assert isinstance(self.data, dict)
        assert isinstance(self.control, dict)

        for node in (self.data.keys()) | set(self.control.keys()):
            var_name, location = node
            assert isinstance(var_name, str)
            func, lineno = location
            assert callable(func)
            assert isinstance(lineno, int)

    def _source(self, node: Node) -> str:
        # Return source line, or ''
        (name, location) = node
        func, lineno = location
        if not func:  # type: ignore
            # No source
            return ''

        try:
            source_lines, first_lineno = inspect.getsourcelines(func)
        except OSError:
            warnings.warn(f"Couldn't find source "
                          f"for {func} ({func.__name__})")
            return ''

        try:
            line = source_lines[lineno - first_lineno].strip()
        except IndexError:
            return ''

        return line

    def source(self, node: Node) -> str:
        """Return the source code for a given node."""
        line = self._source(node)
        if line:
            return line

        (name, location) = node
        func, lineno = location
        code_name = func.__name__

        if code_name.startswith('<'):
            return code_name
        else:
            return f'<{code_name}()>'
        
    def make_graph(self,
                   name: str = "dependencies",
                   comment: str = "Dependencies") -> Digraph:
        return Digraph(name=name, comment=comment,
            graph_attr={
            },
            node_attr={
                'style': 'filled',
                'shape': 'box',
                'fillcolor': self.NODE_COLOR,
                'fontname': self.FONT_NAME
            },
            edge_attr={
                'fontname': self.FONT_NAME
            })
    

    def graph(self, *, mode: str = 'flow') -> Digraph:
        """
        Draw dependencies. `mode` is either
        * `'flow'`: arrows indicate information flow (from A to B); or
        * `'depend'`: arrows indicate dependencies (B depends on A)
        """
        self.validate()

        g = self.make_graph()
        self.draw_dependencies(g, mode)
        self.add_hierarchy(g)
        return g

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> Any:
        """If the object is output in Jupyter, render dependencies as a SVG graph"""
        return self.graph()._repr_mimebundle_(include, exclude)
    
    def all_vars(self) -> Set[Node]:
        """Return a set of all variables (as `var_name`, `location`) in the dependencies"""
        all_vars = set()
        for var in self.data:
            all_vars.add(var)
            for source in self.data[var]:
                all_vars.add(source)

        for var in self.control:
            all_vars.add(var)
            for source in self.control[var]:
                all_vars.add(source)

        return all_vars
    
    def draw_edge(self, g: Digraph, mode: str,
                  node_from: str, node_to: str, **kwargs: Any) -> None:
        if mode == 'flow':
            g.edge(node_from, node_to, **kwargs)
        elif mode == 'depend':
            g.edge(node_from, node_to, dir="back", **kwargs)
        else:
            raise ValueError("`mode` must be 'flow' or 'depend'")

    def draw_dependencies(self, g: Digraph, mode: str) -> None:
        for var in self.all_vars():
            g.node(self.id(var),
                   label=self.label(var),
                   tooltip=self.tooltip(var))

            if var in self.data:
                for source in self.data[var]:
                    self.draw_edge(g, mode, self.id(source), self.id(var))

            if var in self.control:
                for source in self.control[var]:
                    self.draw_edge(g, mode, self.id(source), self.id(var),
                                   style='dashed', color='grey')
                    
    def id(self, var: Node) -> str:
        """Return a unique ID for `var`."""
        id = ""
        # Avoid non-identifier characters
        for c in repr(var):
            if c.isalnum() or c == '_':
                id += c
            if c == ':' or c == ',':
                id += '_'
        return id

    def label(self, var: Node) -> str:
        """Render node `var` using HTML style."""
        (name, location) = var
        source = self.source(var)

        title = html.escape(name)
        if name.startswith('<'):
            title = f'<I>{title}</I>'

        label = f'<B>{title}</B>'
        if source:
            label += (f'<FONT POINT-SIZE="9.0"><BR/><BR/>'
                      f'{html.escape(source)}'
                      f'</FONT>')
        label = f'<{label}>'
        return label

    def tooltip(self, var: Node) -> str:
        """Return a tooltip for node `var`."""
        (name, location) = var
        func, lineno = location
        return f"{func.__name__}:{lineno}"
    
    def add_hierarchy(self, g: Digraph) -> Digraph:
        """Add invisible edges for a proper hierarchy."""
        functions = self.all_functions()
        for func in functions:
            last_var = None
            last_lineno = 0
            for (lineno, var) in functions[func]:
                if last_var is not None and lineno > last_lineno:
                    g.edge(self.id(last_var),
                           self.id(var),
                           style='invis')

                last_var = var
                last_lineno = lineno

        return g
    
    def all_functions(self) -> Dict[Callable, List[Tuple[int, Node]]]:
        """
        Return mapping 
        {`function`: [(`lineno`, `var`), (`lineno`, `var`), ...], ...}
        for all functions in the dependencies.
        """
        functions: Dict[Callable, List[Tuple[int, Node]]] = {}
        for var in self.all_vars():
            (name, location) = var
            func, lineno = location
            if func not in functions:
                functions[func] = []
            functions[func].append((lineno, var))

        for func in functions:
            functions[func].sort()

        return functions
    
    def expand_criteria(self, criteria: List[Criterion]) -> List[Node]:
        """Return list of vars matched by `criteria`."""
        all_vars = []
        for criterion in criteria:
            criterion_var = None
            criterion_func = None
            criterion_lineno = None

            if isinstance(criterion, str):
                criterion_var = criterion
            elif len(criterion) == 2 and callable(criterion[0]):
                criterion_func, criterion_lineno = criterion
            elif len(criterion) == 2 and isinstance(criterion[0], str):
                criterion_var = criterion[0]
                criterion_func, criterion_lineno = criterion[1]
            else:
                raise ValueError("Invalid argument")

            for var in self.all_vars():
                (var_name, location) = var
                func, lineno = location

                name_matches = (criterion_func is None or
                                criterion_func == func or
                                criterion_func.__name__ == func.__name__)

                location_matches = (criterion_lineno is None or
                                    criterion_lineno == lineno)

                var_matches = (criterion_var is None or
                               criterion_var == var_name)

                if name_matches and location_matches and var_matches:
                    all_vars.append(var)

        return all_vars
    
    def backward_slice(self, *criteria: Criterion, 
                       mode: str = 'cd', depth: int = -1):
        """
        Create a backward slice from nodes `criteria`.
        `mode` can contain 'c' (draw control dependencies)
        and 'd' (draw data dependencies) (default: 'cd')
        """
        data = {}
        control = {}
        queue = self.expand_criteria(criteria)  # type: ignore
        seen = set()

        while len(queue) > 0 and depth != 0:
            var = queue[0]
            queue = queue[1:]
            seen.add(var)

            if 'd' in mode:
                # Follow data dependencies
                data[var] = self.data[var]
                for next_var in data[var]:
                    if next_var not in seen:
                        queue.append(next_var)
            else:
                data[var] = set()

            if 'c' in mode:
                # Follow control dependencies
                control[var] = self.control[var]
                for next_var in control[var]:
                    if next_var not in seen:
                        queue.append(next_var)
            else:
                control[var] = set()

            depth -= 1

        return Dependencies(data, control)
    
    def format_var(self, var: Node, current_func: Optional[Callable] = None) -> str:
        """Return string for `var` in `current_func`."""
        name, location = var
        func, lineno = location
        if current_func and (func == current_func or func.__name__ == current_func.__name__):
            return f"{name} ({lineno})"
        else:
            return f"{name} ({func.__name__}:{lineno})"
        
    def __str__(self) -> str:
        """Return string representation of dependencies"""
        self.validate()

        out = ""
        for func in self.all_functions():
            code_name = func.__name__

            if out != "":
                out += "\n"
            out += f"{code_name}():\n"

            all_vars = list(set(self.data.keys()) | set(self.control.keys()))
            all_vars.sort(key=lambda var: var[1][1])

            for var in all_vars:
                (name, location) = var
                var_func, var_lineno = location
                var_code_name = var_func.__name__

                if var_code_name != code_name:
                    continue

                all_deps = ""
                for (source, arrow) in [(self.data, "<="), (self.control, "<-")]:
                    deps = ""
                    for data_dep in source[var]:
                        if deps == "":
                            deps = f" {arrow} "
                        else:
                            deps += ", "
                        deps += self.format_var(data_dep, func)

                    if deps != "":
                        if all_deps != "":
                            all_deps += ";"
                        all_deps += deps

                if all_deps == "":
                    continue

                out += ("    " + 
                        self.format_var(var, func) +
                        all_deps + "\n")

        return out
    
    def repr_var(self, var: Node) -> str:
        name, location = var
        func, lineno = location
        return f"({repr(name)}, ({func.__name__}, {lineno}))"

    def repr_deps(self, var_set: Set[Node]) -> str:
        if len(var_set) == 0:
            return "set()"

        return ("{" +
                ", ".join(f"{self.repr_var(var)}"
                          for var in var_set) +
                "}")

    def repr_dependencies(self, vars: Dependency) -> str:
        return ("{\n        " +
                ",\n        ".join(
                    f"{self.repr_var(var)}: {self.repr_deps(vars[var])}"
                    for var in vars) +
                "}")

    def __repr__(self) -> str:
        """Represent dependencies as a Python expression"""
        # Useful for saving and restoring values
        return (f"Dependencies(\n" +
                f"    data={self.repr_dependencies(self.data)},\n" +
                f" control={self.repr_dependencies(self.control)})")
    
    def code(self, *items: Callable, mode: str = 'cd') -> None:
        """
        List `items` on standard output, including dependencies as comments. 
        If `items` is empty, all included functions are listed.
        `mode` can contain 'c' (draw control dependencies) and 'd' (draw data dependencies)
        (default: 'cd').
        """

        if len(items) == 0:
            items = cast(Tuple[Callable], self.all_functions().keys())

        for i, item in enumerate(items):
            if i > 0:
                print()
            self._code(item, mode)

    def _code(self, item: Callable, mode: str) -> None:
        # The functions in dependencies may be (instrumented) copies
        # of the original function. Find the function with the same name.
        func = item
        for fn in self.all_functions():
            if fn == item or fn.__name__ == item.__name__:
                func = fn
                break

        all_vars = self.all_vars()
        slice_locations = set(location for (name, location) in all_vars)

        source_lines, first_lineno = inspect.getsourcelines(func)

        n = first_lineno
        for line in source_lines:
            line_location = (func, n)
            if line_location in slice_locations:
                prefix = "* "
            else:
                prefix = "  "

            print(f"{prefix}{n:4} ", end="")

            comment = ""
            for (mode_control, source, arrow) in [
                ('d', self.data, '<='),
                ('c', self.control, '<-')
            ]:
                if mode_control not in mode:
                    continue

                deps = ""
                for var in source:
                    name, location = var
                    if location == line_location:
                        for dep_var in source[var]:
                            if deps == "":
                                deps = arrow + " "
                            else:
                                deps += ", "
                            deps += self.format_var(dep_var, item)

                if deps != "":
                    if comment != "":
                        comment += "; "
                    comment += deps

            if comment != "":
                line = line.rstrip() + "  # " + comment

            print_content(line.rstrip(), '.py')
            print()
            n += 1