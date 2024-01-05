import argparse
import multiprocessing as mp
import os
import pickle
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, date
from functools import partial
from itertools import repeat
from pathlib import Path
from pprint import pp
from statistics import mean
from subprocess import Popen, PIPE
from typing import Optional, List

import dill
from dateutil.parser import parse
from pathos.multiprocessing import ProcessingPool


CPU_COUNT = mp.cpu_count()

CMD_GET_ALL_COMMIT_IDS = "git log --pretty=format:\"%h\" --no-merges"
CMD_GET_COMMIT_FILES = lambda commit: "git diff-tree --no-commit-id --name-only -r {cid}".format(cid=commit.cid)
CMD_GET_COMMIT_AUTHOR = lambda commit: "git show -s --format=\"%ae\" {cid}".format(cid=commit.cid)
CMD_GET_COMMIT_DATE = lambda commit: "git show -s --format=\"%ci\" {cid}".format(cid=commit.cid)
CMD_GET_COMMIT_MESSAGE = lambda commit: "git show -s --format=\"%B\" {cid}".format(cid=commit.cid)

COMMIT_DATA_FILE = "commit_data.bin"
PICKLE_PATH = os.path.join(os.path.dirname(__file__), COMMIT_DATA_FILE)
REPO_PATH = "/home/ckjackslack/Projects/aoc2023/"
repo_pathobj = Path(REPO_PATH)
assert repo_pathobj.exists() and repo_pathobj.is_dir()


chomp_quotes = lambda line: line.strip('"')
first = lambda obj: obj[0] if isinstance(obj, list) and len(obj) > 0 else obj
get_week_number = lambda date_obj: date_obj.isocalendar()[1]
get_year_week_pair = lambda c: (c.created.year, get_week_number(c.created.date()))
whole_year = lambda year: lambda c: date(year, 1, 1) <= c.created.date() < date(year + 1, 1, 1)
only_authors = lambda authors: lambda c: any(author == c.author for author in authors)


def display(idx, commit):
    print(f"Commit#{idx}:")
    print(commit.cid, commit.message)
    print()


def iterate_over(commits, *args):
    for fn in filter(callable, args):
        commits = filter(fn, commits)
    for commit in commits:
        yield commit


def has_message(commits, search):
    in_message = lambda c: search.lower() in c.message.lower()
    return len(list(iterate_over(commits, in_message)))


def get_authors(commits, substring):
    authors = set(c.author for c in commits)
    return [a for a in authors if substring in a]


def group_by(commits, prop=None, fn=None):
    if prop is not None:
        get_key = lambda c: getattr(c, prop)
    elif callable(fn):
        get_key = lambda c: fn(c)
    dd = defaultdict(list)
    for commit in commits:
        key = get_key(commit)
        dd[key].append(commit)
    return dd


def do_count(grouped, fn=lambda e: e):
    counter = Counter()
    for key, commits in grouped.items():
        counter.update({fn(key): len(commits)})
    return counter


def show_top_n_commits(commits, stop=None):
    for idx, commit in enumerate(commits, start=1):
        display(idx, commit)
        if stop == idx:
            break


def get_average_count_by_author_in_year(commits, substring, year):
    authors = get_authors(commits, substring)
    filtered_commits = list(
        iterate_over(
            commits,
            whole_year(year),
            only_authors(
                authors,
            ),
        )
    )
    by_week_number = group_by(
        filtered_commits,
        fn=lambda c: get_week_number(c.created.date())
    )
    # by_week_number = {k: len(v) for k, v in by_week_number.items()}
    by_week_number = do_count(by_week_number)
    print(f"Mean number of commits by {authors!r} per week in {year}:", int(
        mean(by_week_number.values()))
    )


def run_cmd(cmd, *args, preprocess_line=None, preprocess_whole=None, **kwargs):
    p = Popen(
        cmd(*args).split()
        if callable(cmd) else cmd.split(),
        stdout=PIPE,
        stderr=PIPE,
    )
    stdout, _ = p.communicate()
    out = stdout.decode().strip().split("\n")
    if callable(preprocess_line):
        out = list(map(preprocess_line, out))
    elif type(preprocess_line) is list and all(map(callable, preprocess_line)):
        for fn in preprocess_line:
            out = list(map(fn, out))
    if callable(preprocess_whole):
        return preprocess_whole(out)
    return out


@dataclass
class Commit:
    cid: str
    created: Optional[datetime] = None
    author: Optional[str] = None
    message: Optional[str] = None
    files: Optional[List[str]] = field(default_factory=list)


@contextmanager
def inside_custom_directory():
    old_cwd = os.getcwd()
    os.chdir(repo_pathobj)
    yield
    os.chdir(old_cwd)


def get_or_create_commits(force=False):
    commits = []
    if force:
        os.remove(PICKLE_PATH)
    if os.path.isfile(PICKLE_PATH):
        with open(PICKLE_PATH, mode="rb") as f:
            commits = pickle.load(f)
    else:
        with inside_custom_directory():
            for commit_id in run_cmd(
                CMD_GET_ALL_COMMIT_IDS,
                preprocess_line=chomp_quotes,
            ):
                commit = Commit(cid=commit_id)
                commits.append(commit)
            
            print(f"Number of commits: {len(commits)}")

            with ProcessingPool(nodes=CPU_COUNT) as pool:
                result = pool.map(
                    partial(run_cmd, CMD_GET_COMMIT_FILES),
                    (c for c in commits),
                )
                for files, commit in zip(result, commits):
                    commit.files = files

            with ProcessingPool(nodes=CPU_COUNT) as pool:
                result = pool.map(partial(run_cmd, CMD_GET_COMMIT_AUTHOR,
                    preprocess_line=chomp_quotes,
                    preprocess_whole=first,
                ), (c for c in commits))
                for author, commit in zip(result, commits):
                    commit.author = author

            with ProcessingPool(nodes=CPU_COUNT) as pool:
                result = pool.map(partial(run_cmd, CMD_GET_COMMIT_DATE,
                    preprocess_line=[chomp_quotes, parse],
                    preprocess_whole=first,
                ), (c for c in commits))
                for _date, commit in zip(result, commits):
                    commit.created = _date

            with ProcessingPool(nodes=CPU_COUNT) as pool:
                result = pool.map(partial(run_cmd, CMD_GET_COMMIT_MESSAGE,
                    preprocess_line=chomp_quotes), (c for c in commits))
                for message, commit in zip(result, commits):
                    commit.message = "\n".join(message).strip()

            with open(PICKLE_PATH, mode="wb") as f:
                pickle.dump(commits, f)

    return commits


def main():
    commits = get_or_create_commits()
    date = datetime.fromtimestamp(
        os.stat(PICKLE_PATH).st_ctime
    ).isoformat().split('T')[0]
    size = len(commits)
    print(f"Number of commits: {size} (as of {date})")

    files = set(*zip(*[c.files for c in commits]))
    files = set(filter(None, files))

    formatter = lambda n, d: f"{n / d * 100 if d else 0:.2f}%"
    edited_specific_extension = lambda ext: lambda c: any(os.path.basename(file).endswith(f".{ext}") for file in c.files)

    for ext in ("sql", "html", "js", "css", "txt", "py"):
        commits_where_modified = len(list(iterate_over(commits, edited_specific_extension(ext))))
        total_in_all_commits = len([f for f in files if f.endswith(ext)])
        print(ext, total_in_all_commits)
        print(f"Percentage of all edited files: {formatter(total_in_all_commits, len(files))}")
        print(f"Percentage of commits where edited: {formatter(commits_where_modified, size)}")

    for commit in commits:
        for file in commit.files:
            files.add(file)
    unique_extensions = sorted(list(filter(None, set(os.path.splitext(file)[1][1:] for file in files))))
    print(unique_extensions)

    for filename in ("views.py", "models.py", "signals.py", "settings.py", "constants.py", "forms.py"):
        print(len(list(iterate_over(commits, lambda c: any(path.endswith(filename) for path in c.files)))))

    by_author = group_by(commits, prop="author")
    for author, their_commits in sorted(by_author.items(), key=lambda t: -len(t[1])):
        print(f"{len(their_commits):>4} {author:>38}")

    for msg in ("", "documentation", "refactor"):
        print(f"Commits that have `{msg}` in message: {has_message(commits, msg)}")
    
    show_top_n_commits(commits, 3)

    first_commit = min(commits, key=lambda c: c.created)
    print(first_commit)
    first_three_commits = sorted(commits, key=lambda c: c.created)[:3]
    pp(first_three_commits)

    get_average_count_by_author_in_year(commits, "ckjackslack", 2023)

    # TODO: add and save diff in Commit, to search in source code

if __name__ == "__main__":
    main()