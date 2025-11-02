"""
dump_fs_index.py

Walk a directory tree (starting at --root) and produce a text file listing all folders
and files for documentation. Supports tree or flat formats, optional hidden files,
and default ignores for venv/.git/node_modules/__pycache__.

Usage examples:
    default: tree format:
    python3 dump_fs_index.py --root . --output project-structure.txt
    python3 dump_fs_index.py --root /path/to/project --output structure.txt --format tree --show-hidden
    Include hidden files:
    python3 dump_fs_index.py -r . -o structure-with-details.txt --show-hidden --details
    Flat listing (each file/dir on its own line):
    python3 dump_fs_index.py -r . -o flat-list.txt --format flat
    Ignore extra folders (e.g., exclude build or dist):
    python3 dump_fs_index.py -r . -o project.txt -i venv .git node_modules build dist

"""

import argparse
from pathlib import Path
import os
import sys
import fnmatch
from datetime import datetime

DEFAULT_IGNORES = ["venv", ".venv", ".git", "node_modules", "__pycache__"]


def is_ignored(path: Path, ignore_patterns):
    # check any path part matches an ignore pattern (exact or wildcard)
    for part in path.parts:
        for pat in ignore_patterns:
            if fnmatch.fnmatch(part, pat):
                return True
    return False


def format_file_line(path: Path, root: Path, show_details: bool):
    rel = path.relative_to(root)
    if not show_details:
        return str(rel)
    stat = path.stat()
    size = stat.st_size
    mtime = datetime.fromtimestamp(stat.st_mtime).isoformat(sep=' ', timespec='seconds')
    return f"{rel}    [{size} bytes]    modified: {mtime}"


def write_flat(root: Path, out_f, ignore_patterns, show_hidden, show_details):
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        cur = Path(dirpath)
        # filter out ignored directories from traversal and listing
        dirnames[:] = [d for d in dirnames if not is_ignored(cur / d, ignore_patterns)]
        for d in list(dirnames):
            p = cur / d
            if (not show_hidden) and p.name.startswith('.'):
                continue
            out_f.write(f"DIR: {p.relative_to(root)}\n")
        for f in filenames:
            p = cur / f
            if (not show_hidden) and p.name.startswith('.'):
                continue
            if is_ignored(p, ignore_patterns):
                continue
            out_f.write("FILE: " + format_file_line(p, root, show_details) + "\n")


def write_tree(root: Path, out_f, ignore_patterns, show_hidden, show_details):
    # We will build a sorted tree traversal to have consistent output
    def tree_lines(current: Path, prefix=""):
        try:
            entries = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return
        # separate directories and files for nicer ordering
        dirs = [p for p in entries if p.is_dir()]
        files = [p for p in entries if p.is_file()]

        # process directories
        for i, d in enumerate(dirs):
            if (not show_hidden) and d.name.startswith('.'):
                continue
            if is_ignored(d, ignore_patterns):
                continue
            is_last = (i == len(dirs) - 1 and not files)
            connector = "└── " if is_last else "├── "
            out_f.write(prefix + connector + str(d.relative_to(root)) + "\n")
            new_prefix = prefix + ("    " if is_last else "│   ")
            tree_lines(d, new_prefix)

        # process files
        for j, f in enumerate(files):
            if (not show_hidden) and f.name.startswith('.'):
                continue
            if is_ignored(f, ignore_patterns):
                continue
            is_last_file = (j == len(files) - 1)
            connector = "└── " if is_last_file else "├── "
            line = prefix + connector + format_file_line(f, root, show_details)
            out_f.write(line + "\n")

    # start with root header
    out_f.write(f"{root.resolve()}\n")
    tree_lines(root, "")


def count_python_loc(root: Path, ignore_patterns, show_hidden):
    """Count total lines in all .py files under the root directory."""
    total_lines = 0
    for dirpath, dirnames, filenames in os.walk(root):
        cur = Path(dirpath)
        dirnames[:] = [d for d in dirnames if not is_ignored(cur / d, ignore_patterns)]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            p = cur / filename
            if (not show_hidden) and p.name.startswith('.'):
                continue
            if is_ignored(p, ignore_patterns):
                continue
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    total_lines += sum(1 for _ in f)
            except Exception as e:
                print(f"Skipped {p}: {e}", file=sys.stderr)
    return total_lines


def main():
    parser = argparse.ArgumentParser(description="Dump a project folder structure to a text file.")
    parser.add_argument("--root", "-r", type=str, default=".", help="Root path to start (default: current dir).")
    parser.add_argument("--output", "-o", type=str, default="project-structure.txt", help="Output text file.")
    parser.add_argument("--format", "-f", choices=["tree", "flat"], default="tree", help="Output format.")
    parser.add_argument("--show-hidden", action="store_true", help="Include hidden files and folders (those starting with '.').")
    parser.add_argument("--ignore", "-i", nargs="*", default=DEFAULT_IGNORES,
                        help="Names or glob patterns to ignore (applies to any path part).")
    parser.add_argument("--details", action="store_true", help="Include file size and modification timestamp for files.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        print(f"Error: root path does not exist or is not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    # Count total Python lines of code
    total_loc = count_python_loc(root, args.ignore, args.show_hidden)

    out_path = Path(args.output).resolve()
    try:
        with open(out_path, "w", encoding="utf-8") as out_f:
            header = (
                f"Project structure dump for: {root}\n"
                f"Generated: {datetime.now().isoformat(sep=' ', timespec='seconds')}\n"
                f"Total lines of code in .py scripts: {total_loc}\n\n"
            )
            out_f.write(header)
            if args.format == "flat":
                write_flat(root, out_f, args.ignore, args.show_hidden, args.details)
            else:
                write_tree(root, out_f, args.ignore, args.show_hidden, args.details)
    except Exception as e:
        print(f"Failed to write output file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote project structure with Python LOC to: {out_path}")


if __name__ == "__main__":
    main()
