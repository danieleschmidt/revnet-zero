"""
API module for RevNet-Zero.

Provides REST API endpoints, request/response models, and
API versioning for external integrations.
"""

from .models import (
    ModelRequest,
    ModelResponse,
    TrainingRequest,
    TrainingResponse,
    ConfigRequest,
    ConfigResponse,
    HealthResponse,
)

from .endpoints import (
    ModelAPI,
    TrainingAPI,
    ConfigAPI,
    HealthAPI,
)

from .versioning import (
    APIVersion,
    VersionManager,
    get_current_version,
    get_supported_versions,
)

__all__ = [
    # Request/Response Models
    'ModelRequest',
    'ModelResponse', 
    'TrainingRequest',
    'TrainingResponse',
    'ConfigRequest',
    'ConfigResponse',
    'HealthResponse',
    
    # API Endpoints
    'ModelAPI',
    'TrainingAPI',
    'ConfigAPI', 
    'HealthAPI',
    
    # Versioning
    'APIVersion',
    'VersionManager',
    'get_current_version',
    'get_supported_versions',
]