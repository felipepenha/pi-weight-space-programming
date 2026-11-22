#!/usr/bin/env python3
"""Standalone ISO SQL:2016 W3C-EBNF loader and LLGuidance validator for out-of-harness evals.

Reads the verbatim `sql-2016.ebnf` file (from https://github.com/ronsavage/SQL/blob/master/sql-2016.ebnf)
without modifying it on disk and lowers W3C EBNF (`::=`) productions deterministically into an
LLGuidance/Outlines-compatible Lark CFG specification.
"""

from __future__ import annotations

from collections import Counter
import os
import re
from typing import Any, Sequence

EXAMPLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EBNF_PATH = os.path.join(EXAMPLE_DIR, "sql-2016.ebnf")

# Compact DFA lexer terminals for ISO SQL:2016 prose-defined lexical rules so
# LLGuidance's Earley parser does not hit item-count limits on character recursion.
SQL_LEXER_TERMINALS: dict[str, str] = {
    "regular_identifier": "/[A-Za-z_][A-Za-z0-9_]*/",
    "SQL_language_identifier": "/[A-Za-z][A-Za-z0-9_]*/",
    "delimited_identifier": r'/"([^"\r\n]|"")+"/',
    "character_string_literal": r"/'([^'\r\n]|'')*'/",
    "national_character_string_literal": r"/[Nn]'([^'\r\n]|'')*'/",
    "binary_string_literal": r"/[Xx]'[0-9A-Fa-f ]*'/",
    "unsigned_integer": "/[0-9]+/",
    "exact_numeric_literal": r"/([0-9]+(\.[0-9]*)?|\.[0-9]+)/",
    "approximate_numeric_literal": r"/([0-9]+(\.[0-9]*)?|\.[0-9]+)[Ee][+-]?[0-9]+/",
    "space": '" "',
    "nondoublequote_character": r'/[^"\r\n]/',
    "nonquote_character": r"/[^'\r\n]/",
    "identifier_start": "/[A-Za-z_]/",
    "identifier_extend": "/[A-Za-z0-9_]/",
}


def load_raw_sql_ebnf(path: str = DEFAULT_EBNF_PATH) -> str:
    """Load the verbatim `sql-2016.ebnf` file from disk."""
    resolved = os.path.abspath(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"SQL:2016 EBNF grammar file not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as f:
        return f.read()


def _normalize_ebnf_literal(lit: str) -> str:
    """Escape W3C EBNF literal to Lark double-quoted string and strip Bottlecaps '*' markers."""
    inner = lit[1:-1]
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_ -]*\*", inner):
        inner = inner[:-1]
    return '"' + inner.replace("\\", "\\\\").replace('"', '\\"') + '"'


def ebnf_to_lark(raw_ebnf: str) -> tuple[str, dict[str, Any]]:
    """Deterministically translate W3C/ISO EBNF (`::=`) into LLGuidance Lark CFG syntax."""
    text = re.sub(r"/\*.*?\*/", "", raw_ebnf, flags=re.S)
    text = re.sub(r"(?m)//[^\n]*$", "", text)
    chunks = re.split(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_-]*)\s*::=", text)
    if len(chunks) < 3:
        raise ValueError("Input EBNF contains no `::=` productions.")

    rules: dict[str, list[str]] = {}
    order: list[str] = []
    for i in range(1, len(chunks), 2):
        name = chunks[i].strip()
        body = " ".join(chunks[i + 1].split())
        if not body:
            continue
        if name not in rules:
            order.append(name)
            rules[name] = [body]
        else:
            rules[name].append(body)

    token_rx = re.compile(r'("[^"]*"|\'[^\']*\'|[A-Za-z_][A-Za-z0-9_.-]*)')
    referenced_or_defined: set[str] = set(order)
    for bs in rules.values():
        for b in bs:
            for m in token_rx.finditer(b):
                tok = m.group(0)
                if not (tok.startswith('"') or tok.startswith("'")):
                    referenced_or_defined.add(tok)

    active_lexer = {
        k: v for k, v in SQL_LEXER_TERMINALS.items() if k in referenced_or_defined
    }

    sym_map: dict[str, str] = {}
    used_lower: set[str] = set()
    for name in order:
        cand = "r_" + re.sub(r"[^a-z0-9_]", "_", name.lower())
        while cand in used_lower:
            cand = f"{cand}_{len(used_lower)}"
        used_lower.add(cand)
        sym_map[name] = cand

    for k in active_lexer:
        if k not in sym_map:
            cand = "r_" + re.sub(r"[^a-z0-9_]", "_", k.lower())
            while cand in used_lower:
                cand = f"{cand}_{len(used_lower)}"
            used_lower.add(cand)
            sym_map[k] = cand
            order.append(k)
            rules[k] = []

    literal_inlines: dict[str, str] = {}
    for name in order:
        if name in active_lexer or len(rules[name]) != 1:
            continue
        b = rules[name][0].strip()
        m = re.fullmatch(r'("[^"]+"|\'[^\']+\')', b)
        if m:
            literal_inlines[name] = _normalize_ebnf_literal(m.group(0))

    def _convert_body(body: str) -> str:
        out: list[str] = []
        pos = 0
        for m in token_rx.finditer(body):
            out.append(body[pos : m.start()])
            tok = m.group(0)
            if tok.startswith('"') or tok.startswith("'"):
                out.append(_normalize_ebnf_literal(tok))
            elif tok in literal_inlines:
                out.append(literal_inlines[tok])
            elif tok in sym_map:
                out.append(sym_map[tok])
            else:
                out.append(
                    '"' + tok.replace("\\", "\\\\").replace('"', '\\"') + '"'
                )
            pos = m.end()
        out.append(body[pos:])
        return "".join(out)

    adj: dict[str, list[str]] = {}
    for name in order:
        if name in active_lexer or name in literal_inlines:
            adj[name] = []
            continue
        refs: list[str] = []
        for m in token_rx.finditer(" ".join(rules[name])):
            tok = m.group(0)
            if (
                not (tok.startswith('"') or tok.startswith("'"))
                and tok in sym_map
                and tok not in literal_inlines
            ):
                refs.append(tok)
        adj[name] = refs

    reachable: set[str] = set()
    stack = [order[0]]
    while stack:
        cur = stack.pop()
        if cur in reachable:
            continue
        reachable.add(cur)
        for nxt in adj.get(cur, []):
            if nxt not in reachable:
                stack.append(nxt)

    lines = [rf"start: /[ \t\r\n]*/ {sym_map[order[0]]} /[ \t\r\n]*/"]
    for name in order:
        if name not in reachable or name in literal_inlines:
            continue
        if name in active_lexer:
            lines.append(f"{sym_map[name]}: {active_lexer[name]}")
            continue
        bodies = rules[name]
        if len(bodies) > 1:
            combined = " | ".join(f"( {_convert_body(b)} )" for b in bodies)
        else:
            combined = _convert_body(bodies[0])
        lines.append(f"{sym_map[name]}: {combined}")
    lines.append(r"%ignore /[ \t\r\n]+/")

    lark_text = "\n".join(lines)
    meta = {
        "entry_rule": order[0],
        "total_rules": len(order),
        "inlined_literals": len(literal_inlines),
        "reachable_rules": len(reachable),
    }
    return lark_text, meta


def build_llg_matcher_spec(
    tokenizer: Any, ebnf_path: str = DEFAULT_EBNF_PATH
) -> tuple[Any, str, str, dict[str, Any]]:
    """Build LLGuidance tokenizer, grammar spec, and Lark text from `sql-2016.ebnf`."""
    import llguidance
    import llguidance.hf

    raw_ebnf = load_raw_sql_ebnf(ebnf_path)
    lark_text, meta = ebnf_to_lark(raw_ebnf)
    llg_tok = llguidance.hf.from_tokenizer(tokenizer)
    g_spec = llguidance.grammar_from("lark", lark_text)
    probe = llguidance.LLMatcher(llg_tok, g_spec)
    if probe.is_error():
        raise ValueError(f"Failed to compile SQL:2016 CFG in LLGuidance: {probe.get_error()}")
    return llg_tok, g_spec, lark_text, meta


def evaluate_sql_grammar_prefix(
    llg_tok: Any,
    g_spec: str,
    tokenizer: Any,
    text: str,
    token_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Consume tokens through `LLMatcher` and report CFG acceptance and valid prefix ratio."""
    import llguidance

    raw_ids = (
        list(token_ids)
        if token_ids is not None
        else tokenizer.encode(text, add_special_tokens=False)
    )
    eos_id = getattr(tokenizer, "eos_token_id", None)
    ids: list[int] = []
    for tid in raw_ids:
        if eos_id is not None and int(tid) == int(eos_id):
            break
        ids.append(int(tid))

    total = len(ids)
    if total == 0:
        return {
            "valid": False,
            "valid_prefix_ratio": 0.0,
            "consumed_tokens": 0,
            "total_tokens": 0,
            "error": "Empty output",
        }

    matcher = llguidance.LLMatcher(llg_tok, g_spec)
    consumed = 0
    for tok_id in ids:
        if not matcher.consume_token(tok_id) or matcher.is_error():
            return {
                "valid": False,
                "valid_prefix_ratio": round(consumed / total, 4),
                "consumed_tokens": consumed,
                "total_tokens": total,
                "error": matcher.get_error(),
            }
        consumed += 1

    accepting = bool(matcher.is_accepting())
    return {
        "valid": accepting,
        "valid_prefix_ratio": 1.0,
        "consumed_tokens": consumed,
        "total_tokens": total,
        "error": None
        if accepting
        else "Statement reached end of output in a non-accepting CFG state",
    }


def is_sql_grammar_valid(
    llg_tok: Any,
    g_spec: str,
    tokenizer: Any,
    text: str,
    token_ids: Sequence[int] | None = None,
) -> tuple[bool, str | None]:
    """Check whether `token_ids` (or encoded `text`) is accepted by the SQL:2016 CFG."""
    res = evaluate_sql_grammar_prefix(
        llg_tok, g_spec, tokenizer, text, token_ids=token_ids
    )
    return bool(res["valid"]), res["error"]


def extract_sql_candidate(raw_output: str) -> str:
    """Extract the last SQL block or statement candidate from a completion."""
    cleaned = re.sub(r"\[/?SQL\]", "", raw_output, flags=re.I).strip()
    fenced = re.findall(r"```(?:sql)?\s*(.*?)```", cleaned, flags=re.I | re.S)
    if fenced:
        return fenced[-1].strip()
    m = re.search(
        r"(?is)\b(?:WITH|SELECT|INSERT|UPDATE|DELETE|CREATE|MERGE|TRUNCATE)\b.*?(?:;|$)",
        cleaned,
    )
    if m:
        return m.group(0).strip()
    return cleaned


_SQL_NON_FUNCTION_WORDS = frozenset(
    {
        "in",
        "exists",
        "values",
        "case",
        "when",
        "then",
        "else",
        "end",
        "and",
        "or",
        "not",
        "where",
        "select",
        "from",
        "as",
        "on",
        "between",
        "like",
        "glob",
        "having",
        "over",
        "filter",
        "within",
        "distinct",
        "all",
        "any",
        "some",
        "cast",
    }
)

_SQL_KEYWORDS = _SQL_NON_FUNCTION_WORDS | frozenset(
    {
        "group",
        "by",
        "order",
        "limit",
        "offset",
        "fetch",
        "join",
        "inner",
        "left",
        "right",
        "full",
        "outer",
        "cross",
        "natural",
        "union",
        "intersect",
        "except",
        "with",
        "is",
        "null",
        "true",
        "false",
        "asc",
        "desc",
        "into",
        "set",
        "update",
        "insert",
        "delete",
        "create",
        "table",
    }
)

_CLAUSE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("IS NOT NULL", r"\bIS\s+NOT\s+NULL\b"),
    ("IS NULL", r"\bIS\s+NULL\b"),
    ("GROUP BY", r"\bGROUP\s+BY\b"),
    ("ORDER BY", r"\bORDER\s+BY\b"),
    ("SELECT", r"\bSELECT\b"),
    ("DISTINCT", r"\bDISTINCT\b"),
    ("FROM", r"\bFROM\b"),
    ("WHERE", r"\bWHERE\b"),
    ("HAVING", r"\bHAVING\b"),
    ("LIMIT", r"\bLIMIT\b"),
    ("FETCH", r"\bFETCH\b"),
    ("JOIN", r"\bJOIN\b"),
    ("UNION", r"\bUNION\b"),
    ("INTERSECT", r"\bINTERSECT\b"),
    ("EXCEPT", r"\bEXCEPT\b"),
    ("WITH", r"\bWITH\b"),
    ("CASE", r"\bCASE\b"),
    ("WHEN", r"\bWHEN\b"),
    ("LIKE", r"\bLIKE\b"),
    ("IN", r"\bIN\b"),
    ("BETWEEN", r"\bBETWEEN\b"),
    ("AND", r"\bAND\b"),
    ("OR", r"\bOR\b"),
    ("NOT", r"\bNOT\b"),
    ("ASC", r"\bASC\b"),
    ("DESC", r"\bDESC\b"),
)


def _find_top_level_keyword(sql: str, keyword_rx: str, start: int = 0) -> tuple[int, int] | None:
    """Find first match of `keyword_rx` at parenthesis depth 0 outside double quotes."""
    rx = re.compile(keyword_rx, flags=re.I)
    depth = 0
    in_dq = False
    i = start
    n = len(sql)
    while i < n:
        ch = sql[i]
        if ch == '"':
            if in_dq and i + 1 < n and sql[i + 1] == '"':
                i += 2
                continue
            in_dq = not in_dq
            i += 1
            continue
        if not in_dq:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif depth == 0 and (i == 0 or not (sql[i - 1].isalnum() or sql[i - 1] == "_")):
                m = rx.match(sql, i)
                if m:
                    return m.start(), m.end()
        i += 1
    return None


def _split_top_level_commas(expr: str) -> list[str]:
    """Split a projection expression list at depth-0 commas outside quotes."""
    parts: list[str] = []
    depth = 0
    in_dq = False
    start = 0
    i = 0
    n = len(expr)
    while i < n:
        ch = expr[i]
        if ch == '"':
            if in_dq and i + 1 < n and expr[i + 1] == '"':
                i += 2
                continue
            in_dq = not in_dq
            i += 1
            continue
        if not in_dq:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                seg = expr[start:i].strip()
                if seg:
                    parts.append(seg)
                start = i + 1
        i += 1
    tail = expr[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def extract_sql_components(sql: str) -> dict[str, Counter[str]]:
    """Extract normalized structural SQL clause/function/identifier multisets for partial-credit F1."""
    components: dict[str, Counter[str]] = {
        "clauses": Counter(),
        "functions": Counter(),
        "identifiers": Counter(),
        "select_items": Counter(),
        "where_ops": Counter(),
    }
    if not sql or not sql.strip():
        return components

    cleaned = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    cleaned = re.sub(r"(?m)--[^\n]*", " ", cleaned)

    # Extract and mask single-quoted string literals first so literal contents never match SQL keywords
    str_literals: list[str] = []

    def _mask_str(m: re.Match[str]) -> str:
        idx = len(str_literals)
        str_literals.append(m.group(0).lower())
        return f" __STR_{idx}__ "

    masked_no_str = re.sub(r"'(?:''|[^'])*'", _mask_str, cleaned)

    # Extract delimited identifiers before masking quotes for structural clause matching
    for raw_id in re.findall(r'"((?:""|[^"])+)"', masked_no_str):
        ident = raw_id.replace('""', '"').strip().lower()
        if ident:
            components["identifiers"][ident] += 1

    for m in re.finditer(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)\b", masked_no_str, flags=re.I):
        cand = m.group(1).lower()
        if not cand.startswith("__str_") and cand not in _SQL_KEYWORDS:
            components["identifiers"][cand] += 1

    structural = re.sub(r'"(?:""|[^"])+"', " __IDENT__ ", masked_no_str)

    # Scan canonical clauses (masking compound clauses so sub-tokens don't double-count)
    clause_scan = structural
    for canon, pat in _CLAUSE_PATTERNS:
        matches = re.findall(pat, clause_scan, flags=re.I)
        if matches:
            components["clauses"][canon] += len(matches)
            clause_scan = re.sub(pat, " ", clause_scan, flags=re.I)

    if components["clauses"]["SELECT"] > 1:
        components["clauses"]["SUBQUERY"] += components["clauses"]["SELECT"] - 1

    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", structural):
        fn = m.group(1).lower()
        if fn not in _SQL_NON_FUNCTION_WORDS and not fn.startswith("__"):
            components["functions"][fn] += 1

    # Extract top-level SELECT items up to top-level FROM
    def _restore_strs(s: str) -> str:
        return re.sub(r"__STR_(\d+)__", lambda m: str_literals[int(m.group(1))], s)

    sel_span = _find_top_level_keyword(masked_no_str, r"SELECT\b(?:\s+(?:DISTINCT|ALL)\b)?")
    if sel_span is not None:
        from_span = _find_top_level_keyword(masked_no_str, r"FROM\b|;", start=sel_span[1])
        end_pos = from_span[0] if from_span is not None else len(masked_no_str)
        proj_body = masked_no_str[sel_span[1] : end_pos]
        for item in _split_top_level_commas(proj_body):
            item_norm = re.sub(
                r'\s+AS\s+(?:"(?:""|[^"])+"|[A-Za-z_][A-Za-z0-9_]*)\s*$',
                "",
                item.strip(),
                flags=re.I,
            )
            item_norm = re.sub(r"\s+", " ", _restore_strs(item_norm)).strip().lower()
            if item_norm:
                components["select_items"][item_norm] += 1

    # Extract WHERE operators, column identifiers, and string literals inside top-level WHERE span
    where_span = _find_top_level_keyword(masked_no_str, r"WHERE\b")
    if where_span is not None:
        end_where = _find_top_level_keyword(
            masked_no_str,
            r"(?:GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|FETCH|UNION|INTERSECT|EXCEPT|;)\b|;",
            start=where_span[1],
        )
        where_body = masked_no_str[
            where_span[1] : (end_where[0] if end_where is not None else len(masked_no_str))
        ]
        where_no_dq = re.sub(
            r'"((?:""|[^"])+)"',
            lambda m: " ",
            where_body,
        )
        for raw_id in re.findall(r'"((?:""|[^"])+)"', where_body):
            ident = raw_id.replace('""', '"').strip().lower()
            if ident:
                components["where_ops"][ident] += 1
        for str_ref in re.findall(r"__STR_(\d+)__", where_no_dq):
            components["where_ops"][str_literals[int(str_ref)]] += 1
        where_ops_scan = re.sub(r"__STR_\d+__", " ", where_no_dq)
        for op_canon, op_pat in (
            ("IS NOT NULL", r"\bIS\s+NOT\s+NULL\b"),
            ("IS NULL", r"\bIS\s+NULL\b"),
            ("BETWEEN", r"\bBETWEEN\b"),
            ("LIKE", r"\bLIKE\b"),
            ("GLOB", r"\bGLOB\b"),
            ("IN", r"\bIN\b"),
            ("!=", r"!="),
            ("<>", r"<>"),
            ("<=", r"<="),
            (">=", r">="),
            ("=", r"="),
            ("<", r"<"),
            (">", r">"),
        ):
            hits = re.findall(op_pat, where_ops_scan, flags=re.I)
            if hits:
                components["where_ops"][op_canon] += len(hits)
                where_ops_scan = re.sub(op_pat, " ", where_ops_scan, flags=re.I)
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", where_ops_scan):
            cand = m.group(1).lower()
            if not cand.startswith("__") and cand not in _SQL_KEYWORDS:
                components["where_ops"][cand] += 1

    return components


def _multiset_f1(pred_bag: Counter[str], gold_bag: Counter[str]) -> float | None:
    """Compute multiset F1 in [0, 1], returning None when both bags are empty."""
    p_tot = sum(pred_bag.values())
    g_tot = sum(gold_bag.values())
    if p_tot == 0 and g_tot == 0:
        return None
    if p_tot == 0 or g_tot == 0:
        return 0.0
    tp = sum((pred_bag & gold_bag).values())
    return round((2.0 * tp) / (p_tot + g_tot), 4)


def compute_sql_component_f1(
    pred_sql: str, gold_sql: str
) -> dict[str, float | None]:
    """Compute overall and per-component AST/clause multiset F1 between predicted and gold SQL."""
    p_comp = extract_sql_components(pred_sql)
    g_comp = extract_sql_components(gold_sql)

    p_all: Counter[tuple[str, str]] = Counter(
        {(cat, tok): cnt for cat, bag in p_comp.items() for tok, cnt in bag.items()}
    )
    g_all: Counter[tuple[str, str]] = Counter(
        {(cat, tok): cnt for cat, bag in g_comp.items() for tok, cnt in bag.items()}
    )
    p_tot = sum(p_all.values())
    g_tot = sum(g_all.values())
    if p_tot == 0 and g_tot == 0:
        overall = 1.0
    elif p_tot == 0 or g_tot == 0:
        overall = 0.0
    else:
        overall = round((2.0 * sum((p_all & g_all).values())) / (p_tot + g_tot), 4)

    return {
        "overall_f1": overall,
        "clauses_f1": _multiset_f1(p_comp["clauses"], g_comp["clauses"]),
        "functions_f1": _multiset_f1(p_comp["functions"], g_comp["functions"]),
        "identifiers_f1": _multiset_f1(p_comp["identifiers"], g_comp["identifiers"]),
        "select_f1": _multiset_f1(p_comp["select_items"], g_comp["select_items"]),
        "where_f1": _multiset_f1(p_comp["where_ops"], g_comp["where_ops"]),
    }
