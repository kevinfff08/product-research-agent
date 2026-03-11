"""Tests for GitHub API client."""

from __future__ import annotations

import pytest
import httpx
import respx

from src.apis.github_client import GitHubClient


@pytest.fixture
def github(tmp_path):
    return GitHubClient(token="test-token", cache_dir=tmp_path / "cache")


@pytest.fixture
def mock_repo():
    return {
        "full_name": "owner/repo",
        "html_url": "https://github.com/owner/repo",
        "stargazers_count": 1000,
        "forks_count": 200,
        "open_issues_count": 50,
        "language": "Python",
        "description": "A test repository",
        "updated_at": "2025-01-01T00:00:00Z",
        "created_at": "2020-01-01T00:00:00Z",
        "license": {"spdx_id": "MIT"},
        "archived": False,
        "topics": ["python", "ai"],
        "size": 5000,
        "default_branch": "main",
        "subscribers_count": 100,
    }


@pytest.mark.asyncio
async def test_search_repos(github, mock_repo):
    with respx.mock:
        respx.get("https://api.github.com/search/repositories").mock(
            return_value=httpx.Response(200, json={"items": [mock_repo]})
        )
        repos = await github.search_repos("AI code review")

    assert len(repos) == 1
    assert repos[0]["full_name"] == "owner/repo"


@pytest.mark.asyncio
async def test_get_repo(github, mock_repo):
    with respx.mock:
        respx.get("https://api.github.com/repos/owner/repo").mock(
            return_value=httpx.Response(200, json=mock_repo)
        )
        repo = await github.get_repo("owner", "repo")

    assert repo["full_name"] == "owner/repo"


@pytest.mark.asyncio
async def test_get_readme(github):
    with respx.mock:
        respx.get("https://api.github.com/repos/owner/repo/readme").mock(
            return_value=httpx.Response(200, text="# My Project\nA test repo.")
        )
        readme = await github.get_readme("owner", "repo")

    assert "My Project" in readme


@pytest.mark.asyncio
async def test_get_readme_not_found(github):
    with respx.mock:
        respx.get("https://api.github.com/repos/owner/repo/readme").mock(
            return_value=httpx.Response(404)
        )
        readme = await github.get_readme("owner", "repo")

    assert readme == ""


def test_parse_repo_url():
    assert GitHubClient.parse_repo_url("https://github.com/owner/repo") == ("owner", "repo")
    assert GitHubClient.parse_repo_url("https://github.com/org/project/") == ("org", "project")
    assert GitHubClient.parse_repo_url("not-a-url") == ("", "")


@pytest.mark.asyncio
async def test_headers_with_token(github):
    headers = github.headers
    assert "Authorization" in headers
    assert headers["Authorization"] == "token test-token"


@pytest.mark.asyncio
async def test_close(github):
    await github.close()
