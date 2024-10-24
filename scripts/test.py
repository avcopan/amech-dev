#!/usr/bin/env python
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import automech
import click
import yaml
from pydantic import BaseModel


class TestConfig(BaseModel):
    name: str


class RepoInfo(BaseModel, extra="allow"):
    autochem: str | list[str]
    autoio: str | list[str]
    autofile: str | list[str]
    mechanalyzer: str | list[str]
    mechdriver: str | list[str]


SRC_DIR = Path("src").resolve()
EXAMPLE_DIR = Path("src/mechdriver/examples").resolve()
TEST_DIR = Path("src/mechdriver/tests").resolve()
TESTS = [
    TestConfig(**c) for c in yaml.safe_load((TEST_DIR / "config.yaml").read_text())
]

SETUP_REPO_INFO_FILE = TEST_DIR / "setup_repo_info.yaml"
SIGNED_REPO_INFO_FILE = TEST_DIR / "signed_repo_info.yaml"


@click.group()
def main():
    """Testing CLI"""
    pass


@main.command("setup")
def setup():
    """Set up the tests to run."""
    for test in TESTS:
        # Remove the test directory, if it exists
        test_dir = TEST_DIR / test.name
        if test_dir.exists():
            print(f"Removing {test_dir}")
            shutil.rmtree(test_dir)

        # Create a clean test directory
        example_inp_dir = EXAMPLE_DIR / test.name / "inp"
        test_inp_dir = test_dir / "inp"
        print(f"Creating {test_inp_dir} from {example_inp_dir}")
        shutil.copytree(example_inp_dir, test_inp_dir, dirs_exist_ok=True)

        # Create the subtasks
        print(f"Setting up subtasks in {test_dir}")
        os.chdir(test_dir)
        automech.subtasks.setup(".", task_groups=("els",))

    # Works, but not yet integrated:
    setup_repo_info = repos_current_version()
    print(f"\nWriting setup repo information to {SETUP_REPO_INFO_FILE}")
    SETUP_REPO_INFO_FILE.write_text(yaml.safe_dump(dict(setup_repo_info)))


@main.command("els")
@click.argument("nodes")
def els(nodes: str):
    """Run electronic structure calculations.

    :param nodes: A comma-separted list of nodes
    """
    # Distributing multiple tests across a list of nodes is not yet set up
    # Currently, it just runs the one test on all of the given nodes
    if len(TESTS) > 1:
        raise NotImplementedError("Not yet set up for multiple tests")

    for test in TESTS:
        test_dir = TEST_DIR / test.name
        subprocess.run(["pixi", "run", "subtasks", "-t", nodes], cwd=test_dir)


@main.command("status")
def status():
    """Check the status of electronic structure calculations."""
    for test in TESTS:
        test_dir = TEST_DIR / test.name
        print(f"Checking status in {test_dir}...")
        os.chdir(test_dir)
        automech.subtasks.status()


@main.command("sign")
def sign():
    """Sign off on electronic structure calculations."""
    setup_repo_info = RepoInfo(**yaml.safe_load(SETUP_REPO_INFO_FILE.read_text()))
    signed_repo_info = repos_version_diff(setup_repo_info)
    print(f"\nWriting signed repo information to {SIGNED_REPO_INFO_FILE}")
    signed_dct = {"username": github_username(), **dict(signed_repo_info)}
    SIGNED_REPO_INFO_FILE.write_text(yaml.safe_dump(signed_dct))


def repos_current_version() -> RepoInfo:
    """Get information about the current version of each repo

    :return: One-line summaries of most recent commits
    """
    repos_warn_about_uncommitted_python_changes()
    version_dct = repos_output(["git", "log", "--oneline", "-n", "1"])
    return RepoInfo(**version_dct)


def repos_version_diff(repo_info: RepoInfo) -> RepoInfo:
    """Get information about differences in the current version of each repo, relative
    to a past version

    :param repo_info: The repository information to compare to
    :return: One-line summaries of each commit since the one given
    """
    repos_warn_about_uncommitted_python_changes()
    command_dct = {
        k: [f"{commit_hash_from_line(v)}..HEAD"] for k, v in dict(repo_info).items()
    }
    diff_dct = repos_output(["git", "log", "--oneline"], command_dct=command_dct)
    diff_dct = {k: v.splitlines() for k, v in diff_dct.items()}
    return RepoInfo(**diff_dct)


def repos_warn_about_uncommitted_python_changes() -> None:
    """Assert that there are no uncommited Python changes in any repo"""
    status_dct = repos_output(["git", "status", "-s", "*.py"])
    for repo, status in status_dct.items():
        if status:
            color_start = "\033[93m"
            color_end = "\033[0m"
            warnings.formatwarning = (
                lambda m, c, *_: f"{color_start}{c.__name__}{color_end}: {m}"
            )
            warnings.warn(f"{repo} has uncommitted Python changes:\n{status}\n")


def commit_hash_from_line(line: str) -> str:
    """Get the commit hash from a one-line log summary

    :param line: A one-line log summary
    :return: The commit hash
    """
    return line.split()[0]


def repos_output(
    command: list[str], command_dct: dict[str, list[str]] | None = None
) -> dict[str, str]:
    """Get output from a command in each repo

    :param command: A command to run for each repo
    :param command_dct: Additional repo-specific commands, by repo name
    :return: One-line summaries of most recent commits
    """
    command_dct = {} if command_dct is None else command_dct
    output_dct = {}
    for repo in RepoInfo.model_fields:
        repo_dir = SRC_DIR / repo
        full_command = command + command_dct.get(repo, [])
        output_dct[repo] = subprocess.check_output(
            full_command, text=True, cwd=repo_dir
        ).strip()
    return output_dct


def github_username() -> str:
    """Return the current user's GitHub username as a string

    Requires the username to be configured:

        git config --global user.name

    :return: The username
    """
    return subprocess.check_output(
        ["git", "config", "--global", "user.name"], text=True
    ).strip()


if __name__ == "__main__":
    main()
