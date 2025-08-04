# RevNet-Zero Deployment Guide

This guide covers deploying RevNet-Zero models in production environments.

## Quick Start

### Basic Model Serving

```bash
# Install with serving dependencies
pip install revnet-zero[full]

# Serve a model
python -m revnet_zero.deployment.serving model.pt --host 0.0.0.0 --port 8000
```

### Docker Deployment

```bash
# Build Docker image
docker build -t revnet-zero-server .

# Run container
docker run -p 8000:8000 -v /path/to/model:/app/model revnet-zero-server
```

## Architecture Overview

RevNet-Zero provides a complete deployment stack:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   API Gateway   │    │     Clients     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────────────────────────────────┐
         │              Model Server Cluster           │
         │  ┌─────────────┐  ┌─────────────┐  ┌──────│
         │  │   Server 1  │  │   Server 2  │  │  ... │
         │  └─────────────┘  └─────────────┘  └──────│
         └─────────────────────────────────────────────┘
                                 │
         ┌─────────────────────────────────────────────┐
         │              Monitoring & Caching           │
         │  ┌─────────────┐  ┌─────────────┐  ┌──────│
         │  │  Prometheus │  │    Redis    │  │ Logs │
         │  └─────────────┘  └─────────────┘  └──────│
         └─────────────────────────────────────────────┘
```

## Model Server Configuration

### Basic Configuration

```python
from revnet_zero.deployment import ServingConfig, ModelServer

config = ServingConfig(
    host="0.0.0.0",
    port=8000,
    max_batch_size=8,
    max_sequence_length=2048,
    cache_enabled=True,
    cache_size_mb=512,
    use_amp=True,
)

server = ModelServer("path/to/model.pt", config)
```

### Advanced Configuration

```python
config = ServingConfig(
    # Server settings
    host="0.0.0.0",
    port=8000,
    workers=4,
    
    # Performance settings
    max_batch_size=16,
    max_sequence_length=4096,
    timeout_seconds=30,
    enable_batching=True,
    batch_timeout_ms=50,
    
    # Memory management
    cache_enabled=True,
    cache_size_mb=1024,
    use_amp=True,
    
    # Monitoring
    enable_metrics=True,
    metrics_port=9090,
    health_check_interval=30,
)
```

## Performance Optimization

### Memory Optimization

```python
# Enable reversible mode for memory efficiency
model.set_reversible_mode(True)

# Configure memory scheduler
from revnet_zero.memory import AdaptiveScheduler
scheduler = AdaptiveScheduler(model, memory_budget=8*1024**3)  # 8GB
model.set_memory_scheduler(scheduler)
```

### Caching Configuration

```python
from revnet_zero.optimization import CacheConfig, CacheManager

cache_config = CacheConfig(
    max_memory_mb=2048,
    max_entries=10000,
    ttl_seconds=3600,
    enable_disk_cache=True,
    cache_hit_threshold=0.1,
)

cache_manager = CacheManager(cache_config)
```

### Performance Profiling

```python
from revnet_zero.optimization import PerformanceProfiler, ProfilerConfig

profiler_config = ProfilerConfig(
    enable_gpu_profiling=True,
    enable_memory_profiling=True,
    sampling_interval=0.1,
    export_traces=True,
)

profiler = PerformanceProfiler(profiler_config)
profiler.start_profiling()

# Run your model
outputs = model(inputs)

profiler.stop_profiling()
summary = profiler.get_performance_summary()
```

## API Reference

### Generation Endpoint

**POST** `/generate`

```json
{
  "input_text": "The future of AI is",
  "max_length": 100,
  "temperature": 0.8,
  "top_p": 0.9,
  "top_k": 50,
  "num_return_sequences": 1,
  "do_sample": true
}
```

Response:
```json
{
  "generated_texts": ["The future of AI is bright..."],
  "request_id": "req_12345",
  "processing_time": 0.245,
  "tokens_generated": 85,
  "model_info": {
    "model_type": "ReversibleTransformer",
    "parameters": 125000000
  }
}
```

### Health Check

**GET** `/health`

```json
{
  "healthy": true,
  "model_loaded": true,
  "device": "cuda:0",
  "memory_usage": {
    "allocated_mb": 1234.5,
    "reserved_mb": 2048.0
  },
  "uptime": 3600.0
}
```

### Metrics

**GET** `/metrics`

```json
{
  "requests_processed": 1000,
  "avg_processing_time": 0.123,
  "total_tokens_generated": 50000,
  "cache_hits": 450,
  "cache_misses": 550,
  "errors": 5
}
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Install RevNet-Zero
RUN pip3 install -e .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["python3", "-m", "revnet_zero.deployment.serving", "model.pt"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  revnet-server:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
  
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

## Cloud Deployment

### AWS ECS

```json
{
  "family": "revnet-zero-task",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "8192",
  "containerDefinitions": [
    {
      "name": "revnet-server",
      "image": "your-registry/revnet-zero:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MODEL_PATH",
          "value": "/app/models/model.pt"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/revnet-zero",
          "awslogs-region": "us-west-2"
        }
      }
    }
  ]
}
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: revnet-zero-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: revnet-zero-server
  template:
    metadata:
      labels:
        app: revnet-zero-server
    spec:
      containers:
      - name: server
        image: revnet-zero:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "1000m"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
        env:
        - name: MODEL_PATH
          value: "/app/models/model.pt"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: revnet-zero-service
spec:
  selector:
    app: revnet-zero-server
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Monitoring and Observability

### Prometheus Metrics

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'revnet-zero'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
```

### Grafana Dashboard

Key metrics to monitor:
- Request latency (p50, p95, p99)
- Throughput (requests/second)
- GPU utilization
- Memory usage
- Cache hit rate
- Error rate

### Logging

```python
import logging
from revnet_zero.utils.logging import setup_logging

# Setup structured logging
logger = setup_logging(
    level="INFO",
    log_dir="/var/log/revnet-zero",
    structured=True
)

# Log requests
logger.info("Processing request", 
           request_id="req_123",
           input_length=50,
           max_length=100)
```

## Security Considerations

### Authentication

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(token: str = Depends(security)):
    # Implement token verification
    if not verify_jwt_token(token.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
```

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/generate")
@limiter.limit("10/minute")
async def generate_endpoint(request: Request, ...):
    # Rate-limited endpoint
    pass
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size
   - Enable reversible mode
   - Increase GPU memory

2. **Slow Inference**
   - Enable AMP (automatic mixed precision)
   - Optimize sequence length
   - Check GPU utilization

3. **High Latency**
   - Enable batching
   - Optimize cache settings
   - Check network latency

### Performance Tuning

```python
# Optimize model for inference
model.eval()
model = torch.jit.trace(model, example_input)

# Enable optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
```

## Best Practices

1. **Model Management**
   - Version your models
   - Use A/B testing for model updates
   - Implement gradual rollouts

2. **Resource Management**
   - Monitor GPU memory usage
   - Implement proper error handling
   - Use circuit breakers

3. **Scalability**
   - Implement horizontal scaling
   - Use load balancing
   - Monitor performance metrics

4. **Reliability**
   - Implement health checks
   - Use redundancy
   - Plan for disaster recovery

## Support

For deployment issues:
1. Check the logs for error messages
2. Verify model compatibility
3. Monitor system resources
4. Contact support with performance metrics

## Examples

See the `examples/deployment/` directory for complete deployment examples including:
- Docker configurations
- Kubernetes manifests  
- Cloud deployment templates
- Monitoring setups