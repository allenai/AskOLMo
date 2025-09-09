from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SafetyCheckRequest:
    content: str


class SafetyCheckResponse(ABC):
    @abstractmethod
    def is_safe(self) -> bool:
        """Return True if the content is safe, False otherwise."""
        raise NotImplementedError

    @abstractmethod
    def get_violation_categories(self) -> list[str]:
        """Return a list of violation categories if any."""
        raise NotImplementedError


class SafetyChecker(ABC):
    @abstractmethod
    def check_request(self, req: SafetyCheckRequest) -> SafetyCheckResponse:
        """Check if the given request is safe."""
        raise NotImplementedError
