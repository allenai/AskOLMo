import os
from dataclasses import dataclass

import yaml


@dataclass
class GoogleCloudServices:
    api_key: str


@dataclass
class Config:
    google_cloud_services: GoogleCloudServices


class ConfigManager:
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.cfg = self._load_config()

    def _load_config(self) -> Config:
        with open(self.config_file, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        google_api_key = data.get("google_cloud", {}).get("api_key", "")
        if not google_api_key:
            google_api_key = os.getenv("GOOGLE_CLOUD_API_KEY", "")

        return Config(google_cloud_services=GoogleCloudServices(api_key=google_api_key))


get_config = ConfigManager()
