from pathlib import Path
import re

def pattern_matching(
    parent_dir:Path,
    pattern: str
):
    pattern_compiled = re.compile(pattern)
    return sorted(
        [
            path
            for path in parent_dir.iterdir()
                if (
                path.is_dir()
                and pattern_compiled.fullmatch(path.name)
            )
        ],
        key = lambda p:p.name,
    )
