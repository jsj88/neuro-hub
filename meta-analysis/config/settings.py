"""
Global settings and configuration for the meta-analysis toolkit.

Settings can be loaded from:
1. Environment variables
2. JSON config file
3. Direct instantiation
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class APISettings:
    """API configuration for LLM and search services."""
    
    # LLM providers
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    default_provider: str = "anthropic"
    default_model: Optional[str] = None
    
    # Search APIs
    pubmed_email: Optional[str] = None
    pubmed_api_key: Optional[str] = None
    serpapi_key: Optional[str] = None
    semantic_scholar_key: Optional[str] = None
    scopus_api_key: Optional[str] = None
    scopus_inst_token: Optional[str] = None


@dataclass
class ProxySettings:
    """Institutional proxy configuration."""
    
    enabled: bool = False
    url: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None


@dataclass
class AnalysisDefaults:
    """Default parameters for meta-analyses."""
    
    # ALE defaults
    ale_kernel_fwhm: Optional[float] = None  # Auto from sample size
    ale_null_method: str = "approximate"
    ale_n_iters: int = 10000
    ale_correction: str = "fwe"
    ale_alpha: float = 0.05
    ale_cluster_threshold: float = 0.001
    
    # Effect size defaults
    es_method: str = "DL"  # DerSimonian-Laird
    es_ci_level: float = 0.95


@dataclass
class PathSettings:
    """Default paths for data and outputs."""
    
    data_dir: str = "data"
    results_dir: str = "results"
    figures_dir: str = "figures"
    cache_dir: str = ".cache"


@dataclass 
class Settings:
    """
    Global settings container.
    
    Example:
        >>> settings = Settings.from_env()
        >>> print(settings.api.anthropic_api_key)
        
        >>> settings = Settings.from_file("config.json")
        >>> settings.save("my_config.json")
    """
    
    api: APISettings = field(default_factory=APISettings)
    proxy: ProxySettings = field(default_factory=ProxySettings)
    analysis: AnalysisDefaults = field(default_factory=AnalysisDefaults)
    paths: PathSettings = field(default_factory=PathSettings)
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls(
            api=APISettings(
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                default_provider=os.getenv("LLM_PROVIDER", "anthropic"),
                pubmed_email=os.getenv("PUBMED_EMAIL"),
                pubmed_api_key=os.getenv("PUBMED_API_KEY"),
                serpapi_key=os.getenv("SERPAPI_KEY"),
                semantic_scholar_key=os.getenv("SEMANTIC_SCHOLAR_KEY"),
                scopus_api_key=os.getenv("SCOPUS_API_KEY"),
                scopus_inst_token=os.getenv("SCOPUS_INST_TOKEN"),
            ),
            proxy=ProxySettings(
                enabled=bool(os.getenv("LIBRARY_PROXY_URL")),
                url=os.getenv("LIBRARY_PROXY_URL"),
                username=os.getenv("LIBRARY_PROXY_USER"),
                password=os.getenv("LIBRARY_PROXY_PASS"),
            ),
            paths=PathSettings(
                data_dir=os.getenv("META_DATA_DIR", "data"),
                results_dir=os.getenv("META_RESULTS_DIR", "results"),
                figures_dir=os.getenv("META_FIGURES_DIR", "figures"),
            )
        )
    
    @classmethod
    def from_file(cls, path: str) -> "Settings":
        """Load settings from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        return cls(
            api=APISettings(**data.get("api", {})),
            proxy=ProxySettings(**data.get("proxy", {})),
            analysis=AnalysisDefaults(**data.get("analysis", {})),
            paths=PathSettings(**data.get("paths", {}))
        )
    
    def save(self, path: str):
        """Save settings to JSON file."""
        data = {
            "api": asdict(self.api),
            "proxy": asdict(self.proxy),
            "analysis": asdict(self.analysis),
            "paths": asdict(self.paths)
        }
        # Remove sensitive keys before saving
        if data["api"].get("anthropic_api_key"):
            data["api"]["anthropic_api_key"] = "***SET_IN_ENV***"
        if data["api"].get("openai_api_key"):
            data["api"]["openai_api_key"] = "***SET_IN_ENV***"
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def ensure_dirs(self):
        """Create output directories if they don't exist."""
        for dir_path in [self.paths.data_dir, self.paths.results_dir, 
                         self.paths.figures_dir, self.paths.cache_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> Dict[str, bool]:
        """Check which services are properly configured."""
        return {
            "anthropic": bool(self.api.anthropic_api_key),
            "openai": bool(self.api.openai_api_key),
            "pubmed": bool(self.api.pubmed_email),
            "serpapi": bool(self.api.serpapi_key),
            "semantic_scholar": True,  # Works without key
            "scopus": bool(self.api.scopus_api_key),
            "proxy": self.proxy.enabled and bool(self.proxy.url)
        }


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance (lazy-loaded from env)."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def configure(settings: Settings):
    """Set global settings instance."""
    global _settings
    _settings = settings
