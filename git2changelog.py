#!/usr/bin/env python3
# Copyright 2008 Marcus D. Hanwell <marcus@cryos.org>
# Distributed under the terms of the GNU General Public License v2 or later

import re
import subprocess
import textwrap

# Message lines that carry no information for the ChangeLog.
IGNORED_MESSAGE_RE = re.compile(r"^(git-svn-id:|Signed-off-by:)")

# A line of --stat output, e.g. "  src/geometry.cpp | 12 ++++++------".
STAT_RE = re.compile(r"^\s+(?P<file>.+?)\s+\|\s+\d+")

# The summary line that ends the --stat block, e.g. " 3 files changed, ...".
STAT_SUMMARY_RE = re.compile(r"^\s+\d+ files? changed")

# Width the commit entries are wrapped to.
WIDTH = 78


def read_git_log():
    """Return the git log output the ChangeLog is generated from."""
    result = subprocess.run(
        ["git", "log", "--summary", "--stat", "--no-merges", "--date=short"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def parse_commits(log):
    """Yield (date, author, message, files) tuples, newest commit first.

    Commits that touch no files are skipped, as they produce no entry.
    """
    date = author = None
    message_lines = []
    files = []
    in_message = False

    def entry():
        if date and author and files:
            return (date, author, " ".join(message_lines), files)
        return None

    for line in log.splitlines():
        # The commit line marks the start of a new commit object.
        if line.startswith("commit "):
            commit = entry()
            if commit:
                yield commit
            date = author = None
            message_lines = []
            files = []
            in_message = False
        elif line.startswith("Author:"):
            author = line.split(":", 1)[1].strip()
        elif line.startswith("Date:"):
            date = line.split(":", 1)[1].strip()
        elif line.startswith("    "):
            # The commit message is the part indented by four spaces.
            in_message = True
            text = line.strip()
            if not IGNORED_MESSAGE_RE.match(text):
                message_lines.append(text)
        elif STAT_SUMMARY_RE.match(line):
            continue
        elif in_message:
            # Collect the files of this commit from the --stat block.
            # FIXME: Still need to add +/- to files
            match = STAT_RE.match(line)
            if match:
                files.append(match.group("file").strip())

    commit = entry()
    if commit:
        yield commit


def write_changelog(commits, out):
    prev_author_line = ""
    for date, author, message, files in commits:
        # The author line is only written if it is the first one for that
        # author on this day.
        author_line = date + " " + author
        if not prev_author_line:
            out.write(author_line + "\n")
        elif author_line != prev_author_line:
            out.write("\n" + author_line + "\n")

        # Assemble the actual commit message line(s) and limit the line length.
        entry = "* " + ", ".join(files) + ": " + message
        for wrapped in textwrap.wrap(entry, width=WIDTH):
            out.write(" " + wrapped + "\n")

        prev_author_line = author_line


def main():
    # Create a ChangeLog file in the current directory.
    with open("ChangeLog", "w") as fout:
        write_changelog(parse_commits(read_git_log()), fout)


if __name__ == "__main__":
    main()
