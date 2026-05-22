"""Configuration management"""

import json
import threading
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict


@dataclass
class MimoAccount:
    """MiMo account config"""
    service_token: str
    user_id: str
    xiaomichatbot_ph: str

    def to_dict(self):
        return asdict(self)


@dataclass
class Config:
    """App config"""
    api_keys: str = "sk-default"
    mimo_accounts: List[MimoAccount] = None

    def __post_init__(self):
        if self.mimo_accounts is None:
            self.mimo_accounts = []

    def to_dict(self):
        return {
            "api_keys": self.api_keys,
            "mimo_accounts": [acc.to_dict() for acc in self.mimo_accounts]
        }


class ConfigManager:
    """Thread-safe config manager"""

    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.config = Config()
        self.lock = threading.RLock()
        self.account_idx = 0
        self.load()

    def load(self):
        """Load config"""
        if not self.config_file.exists():
            self.save()
            return

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                accounts = [
                    MimoAccount(**acc) for acc in data.get('mimo_accounts', [])
                ]
                self.config = Config(
                    api_keys=data.get('api_keys', 'sk-default'),
                    mimo_accounts=accounts
                )
        except Exception as e:
            print(f"Failed to load config: {e}")
            self.config = Config()
            self.save()

    def save(self):
        """Save config"""
        with self.lock:
            try:
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Failed to save config: {e}")

    def validate_api_key(self, key: str) -> bool:
        """Validate API key"""
        with self.lock:
            keys = [k.strip() for k in self.config.api_keys.split(',')]
            return key in keys

    def get_next_account(self) -> Optional[MimoAccount]:
        """Get next account (round-robin)"""
        with self.lock:
            if not self.config.mimo_accounts:
                return None
            account = self.config.mimo_accounts[self.account_idx % len(self.config.mimo_accounts)]
            self.account_idx += 1
            return account

    def update_config(self, new_config: dict):
        """Update config"""
        with self.lock:
            accounts = [
                MimoAccount(**acc) for acc in new_config.get('mimo_accounts', [])
            ]
            self.config = Config(
                api_keys=new_config.get('api_keys', 'sk-default'),
                mimo_accounts=accounts
            )
            self.save()

    def get_config(self) -> dict:
        """Get config"""
        with self.lock:
            return self.config.to_dict()


# Global config manager instance
config_manager = ConfigManager()
