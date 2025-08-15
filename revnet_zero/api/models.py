"""
API request and response models for RevNet-Zero.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum


class ModelType(Enum):
    """Supported model types."""
    REVERSIBLE_TRANSFORMER = "reversible_transformer"
    REVERSIBLE_BERT = "reversible_bert"
    REVERSIBLE_GPT = "reversible_gpt"


class TaskType(Enum):
    """Supported task types."""
    TEXT_GENERATION = "text_generation"
    TEXT_CLASSIFICATION = "text_classification"
    QUESTION_ANSWERING = "question_answering"
    SEQUENCE_LABELING = "sequence_labeling"


@dataclass
class ModelRequest:
    """Request model for inference."""
    text: str
    task_type: TaskType
    max_length: Optional[int] = 512
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    do_sample: bool = True
    return_attention: bool = False
    use_cache: bool = True


@dataclass
class ModelResponse:
    """Response model for inference."""
    generated_text: str
    confidence_score: float
    processing_time_ms: float
    tokens_generated: int
    memory_used_mb: float
    attention_weights: Optional[List[List[float]]] = None


@dataclass
class TrainingRequest:
    """Request model for training."""
    model_type: ModelType
    dataset_path: str
    output_path: str
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 5e-5
    max_seq_length: int = 512
    warmup_steps: int = 0
    save_steps: int = 1000
    eval_steps: int = 500
    use_reversible: bool = True
    memory_scheduler: str = "adaptive"


@dataclass
class TrainingResponse:
    """Response model for training."""
    job_id: str
    status: str  # "started", "running", "completed", "failed"
    current_epoch: int
    current_step: int
    total_steps: int
    current_loss: float
    best_loss: float
    estimated_time_remaining_minutes: float
    memory_usage_mb: float
    logs: List[str]


@dataclass
class ConfigRequest:
    """Request model for configuration."""
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    memory_config: Dict[str, Any]
    deployment_config: Dict[str, Any]


@dataclass
class ConfigResponse:
    """Response model for configuration."""
    config_id: str
    validated: bool
    warnings: List[str]
    errors: List[str]
    estimated_memory_gb: float
    estimated_training_time_hours: float
    recommended_hardware: str


@dataclass
class HealthResponse:
    """Response model for health check."""
    status: str  # "healthy", "degraded", "unhealthy"
    uptime_seconds: float
    version: str
    memory_usage_percent: float
    cpu_usage_percent: float
    gpu_usage_percent: float
    active_requests: int
    total_requests_processed: int
    average_response_time_ms: float
    error_rate_percent: float
    last_check_timestamp: str


@dataclass
class ErrorResponse:
    """Response model for errors."""
    error_code: str
    error_message: str
    error_details: Dict[str, Any]
    timestamp: str
    request_id: str


@dataclass
class BatchModelRequest:
    """Request model for batch inference."""
    texts: List[str]
    task_type: TaskType
    max_length: Optional[int] = 512
    batch_size: int = 8
    return_individual_results: bool = True


@dataclass
class BatchModelResponse:
    """Response model for batch inference."""
    results: List[ModelResponse]
    batch_processing_time_ms: float
    total_tokens_generated: int
    average_confidence_score: float
    batch_id: str


# API Version Models
@dataclass
class APIVersionInfo:
    """Information about API version."""
    version: str
    status: str  # "stable", "beta", "deprecated"
    release_date: str
    deprecation_date: Optional[str]
    breaking_changes: List[str]
    new_features: List[str]


# Metrics and Monitoring Models
@dataclass
class MetricsResponse:
    """Response model for metrics."""
    timestamp: str
    metrics: Dict[str, Union[int, float, str]]
    performance_stats: Dict[str, float]
    resource_usage: Dict[str, float]
    error_stats: Dict[str, int]


@dataclass
class LogEntry:
    """Log entry model."""
    timestamp: str
    level: str  # "DEBUG", "INFO", "WARN", "ERROR"
    message: str
    component: str
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


@dataclass
class LogsResponse:
    """Response model for logs."""
    logs: List[LogEntry]
    total_count: int
    page: int
    page_size: int
    has_more: bool