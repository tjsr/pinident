import sys
import traceback
from collections.abc import Iterator
from types import TracebackType

import pytest


@pytest.fixture(autouse=True)
def fail_on_unhandled_gui_exception(request: pytest.FixtureRequest) -> Iterator[None]:
    """Make wx event exceptions fail the test that dispatched the event."""
    if request.node.get_closest_marker('gui') is None:
        yield
        return
    previous_hook = sys.excepthook
    failures: list[str] = []

    def capture(exc_type: type[BaseException], exc: BaseException,
                tb: TracebackType | None) -> None:
        diagnostic = ''.join(traceback.format_exception(exc_type, exc, tb))
        failures.append(diagnostic)
        sys.__stderr__.write(diagnostic)
        sys.__stderr__.flush()

    sys.excepthook = capture
    try:
        yield
    finally:
        sys.excepthook = previous_hook
        if failures:
            pytest.fail('Unhandled wx event exception:\n' + '\n'.join(failures),
                        pytrace=False)
