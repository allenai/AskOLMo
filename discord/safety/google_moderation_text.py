import logging
from time import time_ns

from google.cloud.language_v2 import (
    Document,
    LanguageServiceClient,
    ModerateTextRequest,
    ModerateTextResponse,
)
from safety.config import get_config
from safety.SafetyChecker import SafetyChecker, SafetyCheckRequest, SafetyCheckResponse


class GoogleModerateTextResponse(SafetyCheckResponse):
    result: ModerateTextResponse
    confidence_threshold = 0.5
    severity_threshold = 0.5
    unsafe_violation_categories = [
        "Toxic",
        "Derogatory",
        "Violent",
        "Sexual",
        "Insult",
        "Profanity",
        "Death, Harm & Tragedy",
        "Firearms & Weapons",
        "Public Safety",
        "War & Conflict",
        "Dangerous Content",
    ]

    def __init__(self, result: ModerateTextResponse):
        self.result = result

    def is_safe(self) -> bool:
        violations = self.get_violation_categories()

        return len(violations) == 0

    def get_violation_categories(self) -> list[str]:
        violations = []

        for category in self.result.moderation_categories:
            if (
                category.name in self.unsafe_violation_categories
                and category.confidence >= self.confidence_threshold
                and category.severity >= self.severity_threshold
            ):
                violations.append(
                    f"<{category.name}> confidence: {category.confidence}; severity: {category.severity}"
                )

        return violations

    def get_scores(self):
        scores = []
        for category in self.result.moderation_categories:
            scores.append(
                {
                    "name": category.name,
                    "confidence": category.confidence,
                    "severity": category.severity,
                }
            )

        return scores


class GoogleModerateText(SafetyChecker):
    client: LanguageServiceClient

    def __init__(self):
        self.client = LanguageServiceClient(
            client_options={"api_key": get_config.cfg.google_cloud_services.api_key}
        )

    def check_request(self, req: SafetyCheckRequest) -> SafetyCheckResponse:
        request = ModerateTextRequest(
            document=Document(content=req.content, type=Document.Type.PLAIN_TEXT),
            model_version="MODEL_VERSION_2",
        )

        start_ns = time_ns()
        result = self.client.moderate_text(request)
        end_ns = time_ns()

        response = GoogleModerateTextResponse(result)

        logging.info(
            {
                "checker": "GoogleModerateText",
                "prompt": req.content,
                "duration_ms": (end_ns - start_ns) / 1_000_000,
                "violations": response.get_violation_categories(),
                "scores": response.get_scores(),
            }
        )

        return response
