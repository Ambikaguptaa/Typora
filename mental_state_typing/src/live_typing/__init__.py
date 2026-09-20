"""Live typing buffer package.

Note: In this foundational phase, live key listening hooks are not implemented.
This package defines the timing event data structures and buffer interfaces.
"""

from src.live_typing.collector import KeystrokeTimingEvent, TimingCollector

__all__ = ["KeystrokeTimingEvent", "TimingCollector"]
