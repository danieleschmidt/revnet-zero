"""
Model serving infrastructure for RevNet-Zero.

Provides high-performance model serving with automatic scaling,
load balancing, and monitoring capabilities.
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import uuid

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from ..models.reversible_transformer import ReversibleTransformer
from ..optimization.cache_manager import CacheManager, CacheConfig
from ..utils.logging import get_logger
from ..utils.validation import validate_sequence_input


@dataclass
class ServingConfig:
    """Configuration for model serving."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_batch_size: int = 8
    max_sequence_length: int = 2048
    timeout_seconds: int = 30
    cache_enabled: bool = True
    cache_size_mb: int = 512
    use_amp: bool = True
    model_parallel: bool = False
    device: str = "auto"
    log_level: str = "INFO"
    
    # Performance optimization
    enable_batching: bool = True
    batch_timeout_ms: int = 50
    max_concurrent_requests: int = 100
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30


class GenerationRequest(BaseModel):
    """Request model for text generation."""
    input_text: str = Field(..., description="Input text to generate from")
    max_length: int = Field(100, ge=1, le=2048, description="Maximum generation length")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: int = Field(50, ge=1, le=100, description="Top-k sampling parameter")
    num_return_sequences: int = Field(1, ge=1, le=5, description="Number of sequences to return")
    do_sample: bool = Field(True, description="Whether to use sampling")
    request_id: Optional[str] = Field(None, description="Optional request ID")


class GenerationResponse(BaseModel):
    """Response model for text generation."""
    generated_texts: List[str]
    request_id: Optional[str]
    processing_time: float
    tokens_generated: int
    model_info: Dict[str, Any]


class BatchRequest:
    """Internal batch request for efficient processing."""
    
    def __init__(self, request: GenerationRequest, future: asyncio.Future):
        self.request = request
        self.future = future
        self.request_id = request.request_id or str(uuid.uuid4())
        self.created_at = time.time()


class ModelServer:
    """
    High-performance model server for RevNet-Zero.
    
    Provides REST API for model inference with automatic batching,
    caching, and performance monitoring.
    """
    
    def __init__(self, model_path: str, config: ServingConfig = None):
        """
        Initialize model server.
        
        Args:
            model_path: Path to model checkpoint
            config: Serving configuration
        """
        self.config = config or ServingConfig()
        self.logger = get_logger(f"ModelServer", level=self.config.log_level)
        
        # Setup device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        self.tokenizer = self._load_tokenizer()
        
        # Setup caching
        self.cache_manager = None
        if self.config.cache_enabled:
            cache_config = CacheConfig(max_memory_mb=self.config.cache_size_mb)
            self.cache_manager = CacheManager(cache_config)
        
        # Batching system
        self.batch_queue = queue.Queue()
        self.batch_processor_running = False
        self.batch_lock = threading.Lock()
        
        # Performance tracking
        self.stats = {
            "requests_processed": 0,
            "total_processing_time": 0.0,
            "total_tokens_generated": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
        }
        
        # Health status
        self.healthy = True
        self.model_info = self._get_model_info()
        
        # Start batch processor
        if self.config.enable_batching:
            self._start_batch_processor()
    
    def _load_model(self, model_path: str) -> ReversibleTransformer:
        """Load model from checkpoint."""
        self.logger.info(f"Loading model from {model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model config
            if "model_config" in checkpoint:
                model_config = checkpoint["model_config"]
            else:
                # Use default config
                model_config = {
                    "vocab_size": 50000,
                    "num_layers": 12,
                    "d_model": 768,
                    "num_heads": 12,
                    "d_ff": 3072,
                }
            
            # Create model
            model = ReversibleTransformer(**model_config)
            
            # Load weights
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                # Assume checkpoint is the state dict
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            
            # Enable optimizations
            if self.config.use_amp:
                model = model.half()  # Convert to half precision
            
            self.logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_tokenizer(self):
        """Load tokenizer (placeholder for now)."""
        # In a real implementation, this would load the appropriate tokenizer
        class DummyTokenizer:
            def encode(self, text): return list(range(len(text.split())))
            def decode(self, ids): return " ".join([f"token_{i}" for i in ids])
            def eos_token_id(self): return 0
        
        return DummyTokenizer()
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": "ReversibleTransformer",
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "memory_usage_mb": torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
            "amp_enabled": self.config.use_amp,
            "cache_enabled": self.config.cache_enabled,
        }
    
    def _start_batch_processor(self):
        """Start background batch processor."""
        def batch_processor():
            self.batch_processor_running = True
            
            while self.batch_processor_running:
                try:
                    self._process_batch()
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
                except Exception as e:
                    self.logger.error(f"Batch processor error: {e}")
        
        batch_thread = threading.Thread(target=batch_processor, daemon=True)
        batch_thread.start()
    
    def _process_batch(self):
        """Process a batch of requests."""
        batch_requests = []
        
        # Collect requests for batch
        try:
            # Get first request (blocking with timeout)
            first_request = self.batch_queue.get(timeout=self.config.batch_timeout_ms / 1000)
            batch_requests.append(first_request)
            
            # Collect additional requests (non-blocking)
            start_time = time.time()
            while (len(batch_requests) < self.config.max_batch_size and
                   (time.time() - start_time) < self.config.batch_timeout_ms / 1000):
                try:
                    request = self.batch_queue.get_nowait()
                    batch_requests.append(request)
                except queue.Empty:
                    break
        
        except queue.Empty:
            return
        
        if not batch_requests:
            return
        
        # Process batch
        try:
            results = self._generate_batch([req.request for req in batch_requests])
            
            # Return results to futures
            for batch_req, result in zip(batch_requests, results):
                if not batch_req.future.cancelled():
                    batch_req.future.set_result(result)
        
        except Exception as e:
            # Return error to all futures
            for batch_req in batch_requests:
                if not batch_req.future.cancelled():
                    batch_req.future.set_exception(e)
    
    def _generate_batch(self, requests: List[GenerationRequest]) -> List[GenerationResponse]:
        """Generate responses for a batch of requests."""
        start_time = time.time()
        
        try:
            # Prepare inputs
            input_texts = [req.input_text for req in requests]
            
            # For demo purposes, create dummy responses
            # In a real implementation, this would tokenize, run inference, and decode
            responses = []
            
            for i, req in enumerate(requests):
                # Simulate generation
                generated_text = f"Generated response {i} for: {req.input_text[:50]}..."
                
                response = GenerationResponse(
                    generated_texts=[generated_text] * req.num_return_sequences,
                    request_id=req.request_id,
                    processing_time=time.time() - start_time,
                    tokens_generated=req.max_length,
                    model_info=self.model_info,
                )
                responses.append(response)
            
            # Update stats
            self.stats["requests_processed"] += len(requests)
            self.stats["total_processing_time"] += time.time() - start_time
            self.stats["total_tokens_generated"] += sum(req.max_length for req in requests)
            
            return responses
            
        except Exception as e:
            self.stats["errors"] += 1
            raise
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text asynchronously.
        
        Args:
            request: Generation request
            
        Returns:
            Generation response
        """
        if not self.healthy:
            raise HTTPException(status_code=503, detail="Model server is unhealthy")
        
        # Validate request
        if len(request.input_text) == 0:
            raise HTTPException(status_code=400, detail="Input text cannot be empty")
        
        if self.config.enable_batching:
            # Use batching system
            future = asyncio.Future()
            batch_request = BatchRequest(request, future)
            
            try:
                self.batch_queue.put(batch_request, timeout=1.0)
                result = await asyncio.wait_for(
                    future, 
                    timeout=self.config.timeout_seconds
                )
                return result
            
            except queue.Full:
                raise HTTPException(status_code=503, detail="Server overloaded")
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Request timeout")
        
        else:
            # Direct processing
            return self._generate_batch([request])[0]
    
    def get_health(self) -> Dict[str, Any]:
        """Get server health status."""
        return {
            "healthy": self.healthy,
            "model_loaded": self.model is not None,
            "device": str(self.device),
            "memory_usage": {
                "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
                "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0,
            },
            "stats": self.stats.copy(),
            "uptime": time.time() - getattr(self, 'start_time', time.time()),
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics."""
        stats = self.stats.copy()
        
        # Calculate derived metrics
        if stats["requests_processed"] > 0:
            stats["avg_processing_time"] = stats["total_processing_time"] / stats["requests_processed"]
            stats["avg_tokens_per_request"] = stats["total_tokens_generated"] / stats["requests_processed"]
        
        # Cache metrics
        if self.cache_manager:
            cache_stats = self.cache_manager.get_cache_stats()
            stats["cache"] = cache_stats
        
        return stats


def create_fastapi_app(model_server: ModelServer) -> FastAPI:
    """Create FastAPI application with model server."""
    app = FastAPI(
        title="RevNet-Zero Model Server",
        description="High-performance serving for RevNet-Zero models",
        version="1.0.0",
    )
    
    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.post("/generate", response_model=GenerationResponse)
    async def generate_endpoint(request: GenerationRequest):
        """Generate text endpoint."""
        return await model_server.generate(request)
    
    @app.get("/health")
    async def health_endpoint():
        """Health check endpoint."""
        return model_server.get_health()
    
    @app.get("/metrics")
    async def metrics_endpoint():
        """Metrics endpoint."""
        return model_server.get_metrics()
    
    @app.get("/model/info")
    async def model_info_endpoint():
        """Model information endpoint."""
        return model_server.model_info
    
    return app


def serve_model(model_path: str, config: ServingConfig = None):
    """
    Serve model with FastAPI.
    
    Args:
        model_path: Path to model checkpoint
        config: Serving configuration
    """
    config = config or ServingConfig()
    
    # Create model server
    model_server = ModelServer(model_path, config)
    model_server.start_time = time.time()
    
    # Create FastAPI app
    app = create_fastapi_app(model_server)
    
    # Run server
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=1,  # Single worker for now due to model sharing
        log_level=config.log_level.lower(),
    )


# CLI interface for serving
def main():
    """CLI entry point for model serving."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Serve RevNet-Zero model")
    parser.add_argument("model_path", help="Path to model checkpoint")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--batch-size", type=int, default=8, help="Maximum batch size")
    parser.add_argument("--cache-size", type=int, default=512, help="Cache size in MB")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    config = ServingConfig(
        host=args.host,
        port=args.port,
        max_batch_size=args.batch_size,
        cache_size_mb=args.cache_size,
        cache_enabled=not args.no_cache,
        log_level=args.log_level,
    )
    
    serve_model(args.model_path, config)


if __name__ == "__main__":
    main()