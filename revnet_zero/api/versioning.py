"""
API versioning support for RevNet-Zero.
"""

from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, date


class APIVersion(Enum):
    """Supported API versions."""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"


@dataclass
class VersionInfo:
    """Information about an API version."""
    version: str
    status: str  # "stable", "beta", "deprecated"
    release_date: date
    deprecation_date: Optional[date]
    breaking_changes: List[str]
    new_features: List[str]
    supported_until: Optional[date]


class VersionManager:
    """Manager for API versioning."""
    
    def __init__(self):
        self.versions = self._initialize_versions()
        self.current_version = APIVersion.V2_0
        self.default_version = APIVersion.V1_0
    
    def _initialize_versions(self) -> Dict[APIVersion, VersionInfo]:
        """Initialize version information."""
        return {
            APIVersion.V1_0: VersionInfo(
                version="1.0",
                status="stable",
                release_date=date(2024, 1, 1),
                deprecation_date=None,
                breaking_changes=[],
                new_features=[
                    "Basic model inference API",
                    "Training job management",
                    "Health monitoring",
                    "Configuration validation"
                ],
                supported_until=date(2025, 12, 31)
            ),
            APIVersion.V1_1: VersionInfo(
                version="1.1",
                status="stable", 
                release_date=date(2024, 6, 1),
                deprecation_date=None,
                breaking_changes=[],
                new_features=[
                    "Batch inference support",
                    "Enhanced error handling",
                    "Metrics endpoint",
                    "Request tracing"
                ],
                supported_until=date(2025, 12, 31)
            ),
            APIVersion.V2_0: VersionInfo(
                version="2.0",
                status="stable",
                release_date=date(2024, 12, 1),
                deprecation_date=None,
                breaking_changes=[
                    "Updated response format for training endpoints",
                    "Renamed configuration parameters",
                    "Modified error response structure"
                ],
                new_features=[
                    "Multi-region deployment support",
                    "Internationalization",
                    "Advanced security features",
                    "Real-time training monitoring",
                    "Model versioning",
                    "A/B testing support"
                ],
                supported_until=date(2026, 12, 31)
            )
        }
    
    def get_version_info(self, version: APIVersion) -> Optional[VersionInfo]:
        """Get information about a specific version."""
        return self.versions.get(version)
    
    def get_supported_versions(self) -> List[str]:
        """Get list of currently supported versions."""
        today = date.today()
        supported = []
        
        for version, info in self.versions.items():
            if info.status != "deprecated" and (not info.supported_until or info.supported_until >= today):
                supported.append(info.version)
        
        return sorted(supported)
    
    def get_latest_version(self) -> str:
        """Get the latest stable version."""
        return self.current_version.value
    
    def get_default_version(self) -> str:
        """Get the default version for new clients."""
        return self.default_version.value
    
    def is_version_supported(self, version: str) -> bool:
        """Check if a version is currently supported."""
        return version in self.get_supported_versions()
    
    def get_migration_info(self, from_version: str, to_version: str) -> Dict[str, List[str]]:
        """Get migration information between versions."""
        try:
            from_ver = APIVersion(from_version)
            to_ver = APIVersion(to_version)
        except ValueError:
            return {"errors": ["Invalid version specified"]}
        
        from_info = self.get_version_info(from_ver)
        to_info = self.get_version_info(to_ver)
        
        if not from_info or not to_info:
            return {"errors": ["Version information not found"]}
        
        migration_info = {
            "breaking_changes": [],
            "new_features": [],
            "recommendations": [],
            "required_changes": []
        }
        
        # Collect breaking changes between versions
        all_versions = [APIVersion.V1_0, APIVersion.V1_1, APIVersion.V2_0]
        start_idx = all_versions.index(from_ver)
        end_idx = all_versions.index(to_ver)
        
        if end_idx > start_idx:
            for i in range(start_idx + 1, end_idx + 1):
                version_info = self.get_version_info(all_versions[i])
                if version_info:
                    migration_info["breaking_changes"].extend(version_info.breaking_changes)
                    migration_info["new_features"].extend(version_info.new_features)
        
        # Add recommendations
        if from_version == "1.0" and to_version == "2.0":
            migration_info["recommendations"] = [
                "Update client libraries to handle new response formats",
                "Review configuration parameter names",
                "Test error handling with new error response structure",
                "Consider using new multi-region features"
            ]
            migration_info["required_changes"] = [
                "Update training endpoint response parsing",
                "Rename deprecated configuration parameters",
                "Handle new error response format"
            ]
        
        return migration_info
    
    def get_deprecation_notices(self) -> List[Dict[str, str]]:
        """Get current deprecation notices."""
        notices = []
        today = date.today()
        
        for version, info in self.versions.items():
            if info.status == "deprecated":
                notices.append({
                    "version": info.version,
                    "message": f"API version {info.version} is deprecated",
                    "deprecation_date": info.deprecation_date.isoformat() if info.deprecation_date else "N/A",
                    "supported_until": info.supported_until.isoformat() if info.supported_until else "N/A"
                })
            elif info.supported_until and info.supported_until < today + datetime.timedelta(days=180):
                notices.append({
                    "version": info.version,
                    "message": f"API version {info.version} will be deprecated soon",
                    "supported_until": info.supported_until.isoformat(),
                    "recommendation": "Plan migration to newer version"
                })
        
        return notices
    
    def validate_version_request(self, requested_version: Optional[str]) -> Dict[str, any]:
        """Validate and normalize version request."""
        if not requested_version:
            return {
                "valid": True,
                "version": self.get_default_version(),
                "message": f"Using default version {self.get_default_version()}"
            }
        
        if not self.is_version_supported(requested_version):
            return {
                "valid": False,
                "version": self.get_default_version(),
                "error": f"Version {requested_version} is not supported",
                "supported_versions": self.get_supported_versions()
            }
        
        return {
            "valid": True,
            "version": requested_version,
            "message": f"Using requested version {requested_version}"
        }


# Global version manager instance
_version_manager = VersionManager()


def get_current_version() -> str:
    """Get current API version."""
    return _version_manager.get_latest_version()


def get_supported_versions() -> List[str]:
    """Get list of supported API versions."""
    return _version_manager.get_supported_versions()


def get_version_manager() -> VersionManager:
    """Get global version manager instance."""
    return _version_manager


def validate_api_version(version: Optional[str]) -> Dict[str, any]:
    """Validate API version request."""
    return _version_manager.validate_version_request(version)


def get_migration_guide(from_version: str, to_version: str) -> Dict[str, List[str]]:
    """Get migration guide between versions."""
    return _version_manager.get_migration_info(from_version, to_version)