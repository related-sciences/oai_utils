import pytest
from _pytest.config import Config
from _pytest.config.argparsing import Parser


def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--openai_api_tests",
        action="store_true",
        default=False,
        help="enables OpenAI API tests",
    )


def pytest_configure(config: Config) -> None:
    config.addinivalue_line(
        "markers", "openai_api_tests: enables OpenAI API tests, requires API key"
    )


def pytest_collection_modifyitems(config, items):
    openai_api_tests_enabled = config.getoption("--openai_api_tests")
    skip_openai_api_tests = pytest.mark.skip(
        reason="OpenAI API tests disabled (--openai_api_tests)"
    )
    for item in items:
        if "openai_api_tests" in item.keywords:
            if not openai_api_tests_enabled:
                item.add_marker(skip_openai_api_tests)
