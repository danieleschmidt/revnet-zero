"""
Production deployment utilities for RevNet-Zero.

This module provides tools for deploying RevNet-Zero models in production
environments including Docker containers, cloud platforms, and edge devices.
"""

from .container import DockerBuilder, create_dockerfile
from .cloud import CloudDeployment, CloudConfig
from .serving import ModelServer, ServingConfig
from .monitoring import DeploymentMonitor, HealthCheck
from .internationalization import (
    InternationalizationManager,
    ComplianceValidator,
    get_i18n_manager,
    set_language,
    get_text,
    is_region_compliant,
    SupportedLanguage,
    RegionCompliance
)
from .multi_region import (
    MultiRegionDeploymentManager,
    RegionConfig,
    Region,
    CloudProvider as MultiRegionCloudProvider,
    LoadBalancingStrategy,
    create_default_multi_region_setup
)

__all__ = [
    "DockerBuilder",
    "create_dockerfile",
    "CloudDeployment", 
    "CloudConfig",
    "ModelServer",
    "ServingConfig",
    "DeploymentMonitor",
    "HealthCheck",
    # Internationalization
    "InternationalizationManager",
    "ComplianceValidator",
    "get_i18n_manager",
    "set_language",
    "get_text",
    "is_region_compliant",
    "SupportedLanguage",
    "RegionCompliance",
    # Multi-Region
    "MultiRegionDeploymentManager",
    "RegionConfig",
    "Region",
    "MultiRegionCloudProvider",
    "LoadBalancingStrategy",
    "create_default_multi_region_setup",
]