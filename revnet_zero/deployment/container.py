"""
Container deployment utilities for RevNet-Zero.

This module provides Docker containerization support for production deployments.
"""

import os
import tempfile
from typing import Dict, List, Optional, Any
from pathlib import Path


class DockerBuilder:
    """Docker container builder for RevNet-Zero models."""
    
    def __init__(self, base_image: str = "python:3.9-slim"):
        self.base_image = base_image
        self.dependencies = [
            "torch>=2.0.0",
            "numpy>=1.21.0", 
            "einops>=0.6.0"
        ]
    
    def build_image(self, model_path: str, image_name: str) -> str:
        """Build Docker image for model deployment."""
        dockerfile = self.create_dockerfile(model_path)
        
        # In a real implementation, this would run docker build
        return f"Built image: {image_name}"
    
    def create_dockerfile(self, model_path: str) -> str:
        """Create Dockerfile for deployment."""
        dockerfile = f"""
FROM {self.base_image}

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and application code
COPY {model_path} ./model/
COPY . .

# Expose port
EXPOSE 8080

# Run server
CMD ["python", "-m", "revnet_zero.deployment.serving"]
"""
        return dockerfile


def create_dockerfile(model_config: Dict[str, Any], output_path: str = "Dockerfile") -> str:
    """Create a Dockerfile for RevNet-Zero deployment."""
    builder = DockerBuilder()
    dockerfile = builder.create_dockerfile(model_config.get("model_path", "./model"))
    
    with open(output_path, "w") as f:
        f.write(dockerfile)
    
    return output_path


def create_deployment_manifest(
    model_name: str,
    image_name: str,
    replicas: int = 3,
    resource_limits: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Create Kubernetes deployment manifest."""
    
    if resource_limits is None:
        resource_limits = {
            "cpu": "1000m",
            "memory": "2Gi", 
            "nvidia.com/gpu": "1"
        }
    
    manifest = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": f"{model_name}-deployment"
        },
        "spec": {
            "replicas": replicas,
            "selector": {
                "matchLabels": {
                    "app": model_name
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": model_name
                    }
                },
                "spec": {
                    "containers": [{
                        "name": model_name,
                        "image": image_name,
                        "ports": [{
                            "containerPort": 8080
                        }],
                        "resources": {
                            "limits": resource_limits,
                            "requests": {
                                "cpu": "500m",
                                "memory": "1Gi"
                            }
                        },
                        "env": [{
                            "name": "MODEL_NAME",
                            "value": model_name
                        }]
                    }]
                }
            }
        }
    }
    
    return manifest


__all__ = ["DockerBuilder", "create_dockerfile", "create_deployment_manifest"]