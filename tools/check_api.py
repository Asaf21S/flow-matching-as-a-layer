import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def module_path(dotted):
    """Resolve a dotted module name to a file inside the repo, or None."""
    parts = dotted.split(".")
    candidate = ROOT.joinpath(*parts).with_suffix(".py")
    package = ROOT.joinpath(*parts, "__init__.py")
    if candidate.is_file():
        return candidate
    if package.is_file():
        return package
    return None


def parse(path):
    """Parse one source file into an AST."""
    return ast.parse(path.read_text(encoding="utf-8"))


def top_level_names(tree):
    """Names a module binds at module level."""
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def signature(node):
    """Summarise a function definition for arity checking."""
    arguments = node.args
    positional = [argument.arg for argument in arguments.posonlyargs + arguments.args]
    return {
        "positional": positional,
        "num_defaults": len(arguments.defaults),
        "has_vararg": arguments.vararg is not None,
        "has_kwarg": arguments.kwarg is not None,
        "kwonly": [argument.arg for argument in arguments.kwonlyargs],
        "kwonly_required": [
            argument.arg
            for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults)
            if default is None
        ],
    }


def collect_definitions():
    """Map ``module.function`` to its signature for every module-level function in src."""
    definitions = {}
    for path in sorted(SRC.rglob("*.py")):
        dotted = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        for node in parse(path).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions[f"{dotted}.{node.name}"] = signature(node)
    return definitions


def check_call(name, call, spec):
    """Return a problem string when a call cannot match the definition."""
    if any(isinstance(argument, ast.Starred) for argument in call.args):
        return None
    if any(keyword.arg is None for keyword in call.keywords):
        return None

    positional = len(call.args)
    keywords = [keyword.arg for keyword in call.keywords]
    allowed = spec["positional"] + spec["kwonly"]

    if not spec["has_vararg"] and positional > len(spec["positional"]):
        return f"{name}: {positional} positional args, takes {len(spec['positional'])}"

    if not spec["has_kwarg"]:
        for keyword in keywords:
            if keyword not in allowed:
                return f"{name}: unexpected keyword {keyword!r}"

    bound = set(spec["positional"][:positional]) | set(keywords)
    required = spec["positional"][: len(spec["positional"]) - spec["num_defaults"]]
    missing = [argument for argument in required if argument not in bound]
    if missing:
        return f"{name}: missing {', '.join(missing)}"

    missing_kwonly = [argument for argument in spec["kwonly_required"] if argument not in keywords]
    if missing_kwonly:
        return f"{name}: missing keyword-only {', '.join(missing_kwonly)}"

    duplicated = [keyword for keyword in keywords if keyword in spec["positional"][:positional]]
    if duplicated:
        return f"{name}: {', '.join(duplicated)} given twice"
    return None


def main():
    definitions = collect_definitions()
    problems = []

    for path in sorted(SRC.rglob("*.py")):
        tree = parse(path)
        relative = path.relative_to(ROOT)
        imported = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src."):
                target = module_path(node.module)
                if target is None:
                    problems.append(f"{relative}: no module {node.module}")
                    continue
                available = top_level_names(parse(target))
                for alias in node.names:
                    if alias.name != "*" and alias.name not in available:
                        problems.append(f"{relative}: {node.module} has no {alias.name!r}")
                    imported[alias.asname or alias.name] = f"{node.module}.{alias.name}"

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            qualified = imported.get(node.func.id)
            spec = definitions.get(qualified) if qualified else None
            if spec is None:
                continue
            problem = check_call(node.func.id, node, spec)
            if problem:
                problems.append(f"{relative}:{node.lineno}: {problem}")

    for problem in problems:
        print("PROBLEM", problem)
    print(f"\n{len(problems)} problem(s) across {len(definitions)} tracked functions.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
