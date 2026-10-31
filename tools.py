"""Tools for the Mech Design Agent: each tool is a plain Python function + a JSON schema."""
import ast
import json
import math
import operator
import pathlib

REPORT_DIR = pathlib.Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

# ---------- 1. calculator (safe, no eval) ----------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_NAMES = {"pi": math.pi, "e": math.e}
_FUNCS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
          "log": math.log, "log10": math.log10, "radians": math.radians, "degrees": math.degrees}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("unsupported expression")


def calculator(expression: str) -> dict:
    result = _eval(ast.parse(expression, mode="eval").body)
    return {"expression": expression, "result": result}


# ---------- 2. material lookup ----------
MATERIALS = {  # typical values; yield/UTS in MPa, density kg/m3, E in GPa
    "mild steel": {"yield": 250, "uts": 410, "density": 7850, "E": 200},
    "en8": {"yield": 465, "uts": 700, "density": 7850, "E": 200},
    "en24": {"yield": 680, "uts": 850, "density": 7850, "E": 210},
    "al6061-t6": {"yield": 276, "uts": 310, "density": 2700, "E": 69},
    "ss304": {"yield": 215, "uts": 505, "density": 8000, "E": 193},
    "ti-6al-4v": {"yield": 880, "uts": 950, "density": 4430, "E": 114},
}


def material_properties(material: str) -> dict:
    key = material.strip().lower()
    if key not in MATERIALS:
        return {"error": f"unknown material '{material}'", "available": list(MATERIALS)}
    return {"material": key, **MATERIALS[key],
            "units": "yield/uts in MPa, density kg/m3, E in GPa"}


# ---------- 3. shaft diameter (torsion only, max shear stress theory) ----------
STD_SIZES = [10, 12, 14, 16, 18, 20, 22, 25, 28, 30, 32, 35, 38, 40, 45,
             50, 55, 60, 65, 70, 75, 80, 90, 100]


def shaft_diameter(power_kw: float, rpm: float, material: str, safety_factor: float = 2.0) -> dict:
    props = material_properties(material)
    if "error" in props:
        return props
    torque_nmm = 9.5493e6 * power_kw / rpm            # T = 9.5493e6 * P / N  [N.mm]
    tau_allow = 0.5 * props["yield"] / safety_factor  # MPa
    d = (16 * torque_nmm / (math.pi * tau_allow)) ** (1 / 3)
    std = next((s for s in STD_SIZES if s >= d), None)
    return {
        "torque_Nm": round(torque_nmm / 1000, 2),
        "allowable_shear_MPa": round(tau_allow, 1),
        "min_diameter_mm": round(d, 2),
        "standard_diameter_mm": std,
        "assumptions": "pure torsion, solid round shaft, tau_allow = 0.5*Sy/SF, no bending/keyway",
    }


# ---------- 4. unit converter ----------
_UNITS = {  # factor to base unit
    "length": {"m": 1, "mm": 1e-3, "cm": 1e-2, "in": 0.0254, "ft": 0.3048},
    "force": {"n": 1, "kn": 1e3, "kgf": 9.80665, "lbf": 4.44822},
    "pressure": {"pa": 1, "kpa": 1e3, "mpa": 1e6, "bar": 1e5, "psi": 6894.76},
    "power": {"w": 1, "kw": 1e3, "hp": 745.7, "ps": 735.5},
    "torque": {"nm": 1, "nmm": 1e-3, "kgfm": 9.80665, "lbfft": 1.35582},
}


def unit_convert(value: float, from_unit: str, to_unit: str) -> dict:
    f, t = from_unit.lower(), to_unit.lower()
    for cat, table in _UNITS.items():
        if f in table and t in table:
            return {"result": value * table[f] / table[t], "category": cat}
    return {"error": "unsupported unit pair", "supported": {k: list(v) for k, v in _UNITS.items()}}


# ---------- 5. save report ----------
def save_report(filename: str, content: str) -> dict:
    path = REPORT_DIR / pathlib.Path(filename).name  # block path traversal
    path.write_text(content, encoding="utf-8")
    return {"saved": str(path), "chars": len(content)}


# ---------- registry + JSON schemas (this is what the LLM sees) ----------
REGISTRY = {
    "calculator": calculator,
    "material_properties": material_properties,
    "shaft_diameter": shaft_diameter,
    "unit_convert": unit_convert,
    "save_report": save_report,
}

TOOL_SCHEMAS = [
    {"name": "calculator",
     "description": "Evaluate a math expression. Supports + - * / ** %, pi, e, sqrt, sin, cos, tan, log, log10.",
     "input_schema": {"type": "object",
                      "properties": {"expression": {"type": "string"}},
                      "required": ["expression"]}},
    {"name": "material_properties",
     "description": "Get yield, UTS, density and Young's modulus for a material "
                    "(mild steel, en8, en24, al6061-t6, ss304, ti-6al-4v).",
     "input_schema": {"type": "object",
                      "properties": {"material": {"type": "string"}},
                      "required": ["material"]}},
    {"name": "shaft_diameter",
     "description": "Minimum solid shaft diameter for a given power and speed under pure torsion.",
     "input_schema": {"type": "object",
                      "properties": {"power_kw": {"type": "number"},
                                     "rpm": {"type": "number"},
                                     "material": {"type": "string"},
                                     "safety_factor": {"type": "number", "default": 2.0}},
                      "required": ["power_kw", "rpm", "material"]}},
    {"name": "unit_convert",
     "description": "Convert between units of length, force, pressure, power or torque.",
     "input_schema": {"type": "object",
                      "properties": {"value": {"type": "number"},
                                     "from_unit": {"type": "string"},
                                     "to_unit": {"type": "string"}},
                      "required": ["value", "from_unit", "to_unit"]}},
    {"name": "save_report",
     "description": "Save the final design report as a markdown file in the reports/ folder.",
     "input_schema": {"type": "object",
                      "properties": {"filename": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["filename", "content"]}},
]


def run_tool(name: str, args: dict) -> str:
    """Execute a tool call; always return a string so errors go back to the LLM instead of crashing."""
    try:
        return json.dumps(REGISTRY[name](**args))
    except Exception as exc:  # the agent can read this and self-correct
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
