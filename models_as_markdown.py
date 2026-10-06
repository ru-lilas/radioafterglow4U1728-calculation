"""build_input.pyのindex.csvを端末の表で確認する。

例:
    python index_to_markdown.py
    python index_to_markdown.py --input data/index.csv --output reports/runs.md
    python index_to_markdown.py --compare f_e

端末表示にはrichが必要。--output指定時はMarkdownも保存する。
端末ではページ送りで表示する（--no-pagerで無効化）。
CSVの値は丸めずに表示する。
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console

GROUPS = {
    "run情報": {
        "run_id": "run ID",
        "run_name": "run名",
        "created_at": "生成日時",
        "config_hash": "設定ハッシュ",
        "input_path": "入力ファイル",
    },
    "モデルパラメータ": {
        "synchrotron_model": "シンクロトロンモデル",
        "eps_th": "熱電子のエネルギー比 eps_th",
        "eps_b": "磁場のエネルギー比 eps_b",
        "mu": "平均分子量 mu",
        "mu_e": "電子の平均分子量 mu_e",
        "distance": "距離",
        "distance_unit": "距離の単位",
        "p_index": "power-law指数 p_index",
        "f_e": "power-law電子のエネルギー比 f_e",
    },
    "パラメータ格子": {
        "phi_th_start": "phi_th: start",
        "phi_th_stop": "phi_th: stop",
        "phi_th_num": "phi_th: 点数",
        "phi_th_scale": "phi_th: scale",
        "phi_th_unit": "phi_th: 単位",
        "theta0_start": "theta0: start",
        "theta0_stop": "theta0: stop",
        "theta0_num": "theta0: 点数",
        "theta0_scale": "theta0: scale",
    },
    "観測・χ²評価": {
        "observation_frequencies": "観測周波数",
        "nu_unit": "周波数の単位",
        "time_min": "時間窓: min",
        "time_max": "時間窓: max",
        "time_unit": "時間の単位",
        "burster_name": "天体名",
    },
}


def escape_markdown(value: str) -> str:
    """HTML・Markdown記号を文字として表示し、表の改行を処理する。"""
    text = html.escape(value, quote=False).replace("\\", "\\\\")
    for char in ("|", "`", "*", "_", "[", "]", "#"):
        text = text.replace(char, "\\" + char)
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def read_index(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames
        if fields is None:
            raise ValueError("fieldsがNoneです")
        if not fields or any(not field for field in fields):
            raise ValueError("index.csvに有効なヘッダーがありません")
        if len(set(fields)) != len(fields):
            raise ValueError("index.csvの列名が重複しています")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"CSVの{reader.line_num}行目の列数がヘッダーと一致しません")
            rows.append(row)
    return list(fields), rows


def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(escape_markdown(item) for item in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(escape_markdown(item) if item else "—" for item in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def timewindow(row: dict[str, str]) -> str:
    start, stop = row.get("time_min", ""), row.get("time_max", "")
    if not start and not stop:
        return ""
    return f"{start or '—'} ～ {stop or '—'} {row.get('time_unit', '')}".strip()


def frequencies(row: dict[str, str]) -> str:
    value = row.get("observation_frequencies", "")
    if not value:
        return ""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        value = ", ".join(str(item) for item in parsed)
    return f"{value} {row.get('nu_unit', '')}".strip()


SUMMARY_HEADERS = ["run", "eps_th", "eps_b", "p_index", "f_e", "観測周波数", "評価時間窓"]


def summary_rows(rows: list[dict[str, str]]) -> list[list[str]]:
    return [
        [row.get("run_name") or row.get("run_id", ""),
         row.get("eps_th", ""), row.get("eps_b", ""), row.get("p_index", ""),
         row.get("f_e", ""), frequencies(row), timewindow(row)]
        for row in rows
    ]


def detail_rows(fields: list[str], row: dict[str, str]) -> list[list[str]]:
    """値と単位、格子設定、関連する値をまとめる。未定義の列も表示する。"""
    combined = [
        ("phi_th", {"phi_th_start": "start", "phi_th_stop": "stop",
                    "phi_th_num": "num", "phi_th_scale": "scale"}, "phi_th_unit"),
        ("theta0", {"theta0_start": "start", "theta0_stop": "stop",
                    "theta0_num": "num", "theta0_scale": "scale"}, None),
        ("molecular_weight", {"mu": "mu", "mu_e": "mu_e"}, None),
        ("distance", {"distance": ""}, "distance_unit"),
        ("powerlaw", {"p_index": "p_index", "f_e": "f_e"}, None),
        ("observation_frequencies", {"observation_frequencies": ""}, "nu_unit"),
        ("timewindow", {"time_min": "min", "time_max": "max"}, "time_unit"),
    ]
    entries: list[list[str]] = []
    used: set[str] = set()
    for group, labels in GROUPS.items():
        for field, label in labels.items():
            if field not in fields or field in used:
                continue
            match = next((item for item in combined if field in item[1]), None)
            if match is None:
                entries.append([group, label, row[field], ""])
                used.add(field)
                continue
            name, components, unit_field = match
            values = [
                f"{short}={row[key] or '—'}" if short else row[key]
                for key, short in components.items() if key in fields
            ]
            unit = row.get(unit_field, "") if unit_field and unit_field in fields else ""
            entries.append([group, name, ", ".join(values), unit])
            used.update(key for key in components if key in fields)
            if unit_field in fields:
                used.add(unit_field)
    entries.extend(["その他", field, row[field], ""] for field in fields if field not in used)
    return entries


RUN_FIELDS = {"run_id", "run_name", "created_at", "config_hash", "input_path"}
DETAIL_HEADERS = ["グループ", "パラメータ", "現在値", "単位"]


def comparison_groups(
    fields: list[str], rows: list[dict[str, str]], parameter: str,
) -> tuple[list[str], list[list[dict[str, str]]]]:
    """CSVの設定列を照合し、指定列だけが異なるrunをグループ化する。"""
    if parameter not in fields:
        raise ValueError(f"比較する列がありません: {parameter}（列名: {', '.join(fields)}）")
    if parameter in RUN_FIELDS:
        raise ValueError(f"run情報ではなくパラメータの列を指定してください: {parameter}")
    common_fields = [field for field in fields if field not in RUN_FIELDS and field != parameter]
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(row[field] for field in common_fields)
        grouped.setdefault(key, []).append(row)
    groups = [
        group for group in grouped.values()
        if len({row[parameter] for row in group}) > 1
    ]
    return common_fields, groups


def print_details(console: Console, title: str, entries: list[list[str]]) -> None:
    from rich.table import Table
    from rich.text import Text

    details = Table(title=Text(title))
    details.add_column("グループ", style="cyan")
    details.add_column("パラメータ")
    details.add_column("現在値", ratio=2)
    details.add_column("単位", no_wrap=True)
    for j, (group, parameter, value, unit) in enumerate(entries):
        shown_group = group if j == 0 or entries[j - 1][0] != group else ""
        end_section = j == len(entries) - 1 or entries[j + 1][0] != group
        details.add_row(
            Text(shown_group), Text(parameter), Text(value or "—"), Text(unit or "—"),
            end_section=end_section,
        )
    console.print(details)


def print_comparison(console: Console, fields: list[str], rows: list[dict[str, str]], parameter: str) -> None:
    from rich.table import Table
    from rich.text import Text

    common_fields, groups = comparison_groups(fields, rows, parameter)
    console.print(Text(f"対照比較: {parameter}（{len(groups)} groups）"))
    console.print("CSVに記録されたほかの設定が同じrunを抽出しています。")
    if not groups:
        console.print("指定パラメータだけが異なるrunはありません。")
    for i, group in enumerate(groups, start=1):
        comparison = Table(title=f"比較グループ {i}")
        comparison.add_column("run")
        comparison.add_column(Text(parameter))
        for row in group:
            comparison.add_row(
                Text(row.get("run_name") or row.get("run_id", "—")),
                Text(row[parameter] or "—"),
            )
        console.print(comparison)
        if common_fields:
            print_details(console, "共通設定", detail_rows(common_fields, group[0]))


def print_index(
    fields: list[str], rows: list[dict[str, str]], *, paginate: bool = True,
    compare: str | None = None,
) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    console = Console()
    use_pager = paginate and console.is_terminal and sys.stdin.isatty()
    if compare is not None:
        comparison_groups(fields, rows, compare)
    # 一覧と全runの詳細を1つのページ送り画面で表示する。
    with console.pager() if use_pager else nullcontext():
        if compare is not None:
            print_comparison(console, fields, rows, compare)
            return
        summary = Table(title=f"run一覧（{len(rows)} runs）")
        for header in SUMMARY_HEADERS:
            summary.add_column(header)
        for values in summary_rows(rows):
            summary.add_row(*(Text(value or "—") for value in values))
        console.print(summary)
        if not rows:
            console.print("登録されたrunはありません。")
            return
        for i, row in enumerate(rows, start=1):
            name = row.get("run_name") or row.get("run_id") or f"行{i}"
            print_details(console, name, detail_rows(fields, row))


def to_markdown(
    fields: list[str], rows: list[dict[str, str]], *, compare: str | None = None,
) -> str:
    if compare is not None:
        common_fields, groups = comparison_groups(fields, rows, compare)
        parts = [f"# 対照比較: {escape_markdown(compare)}", f"比較グループ数: {len(groups)}",
                 "CSVに記録されたほかの設定が同じrunを抽出しています。"]
        if not groups:
            parts.append("指定パラメータだけが異なるrunはありません。")
        for i, group in enumerate(groups, start=1):
            parts.extend([f"## 比較グループ {i}", table(["run", compare], [
                [row.get("run_name") or row.get("run_id", ""), row[compare]] for row in group
            ])])
            if common_fields:
                parts.extend(["### 共通設定", table(DETAIL_HEADERS, detail_rows(common_fields, group[0]))])
        return "\n\n".join(parts) + "\n"
    parts = ["# run一覧", f"run数: {len(rows)}"]
    if not rows:
        return "\n\n".join([*parts, "登録されたrunはありません。", ""])
    parts.append(table(SUMMARY_HEADERS, summary_rows(rows)))
    parts.extend(["# runごとの詳細", "値はCSVに記録された表記を保持しています。記録のない項目は追加しません。"])
    for i, row in enumerate(rows, start=1):
        name = row.get("run_name") or row.get("run_id") or f"行{i}"
        parts.append(f"## {escape_markdown(name)}")
        parts.append(table(DETAIL_HEADERS, detail_rows(fields, row)))
    return "\n\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/index.csv"))
    parser.add_argument("--output", type=Path, help="指定時は表の内容をMarkdownにも保存")
    parser.add_argument("--no-pager", action="store_true", help="ページ送りせず通常出力")
    parser.add_argument("--compare", metavar="COLUMN", help="指定列だけが異なるrunを対照比較（例: f_e）")
    args = parser.parse_args()
    output = args.output
    try:
        if output is not None and output.resolve() == args.input.resolve():
            raise ValueError("入力CSVと出力Markdownには別のパスを指定してください")
        fields, rows = read_index(args.input)
        print_index(fields, rows, paginate=not args.no_pager, compare=args.compare)
        if output is not None:
            markdown = to_markdown(fields, rows, compare=args.compare)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(markdown, encoding="utf-8")
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"エラー: {error}\n")
    if output is not None:
        print(f"生成しました: {output} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
