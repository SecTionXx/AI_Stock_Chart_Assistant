"""
AI Stock Chart Assistant - API Client Module
Robust API client with connection monitoring, retry logic, and health checks.
"""

import time
import asyncio
import threading
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import google.generativeai as genai

from ..core.config import get_config, get_logger
from ..core.error_handler import get_error_handler, handle_exception, ErrorCategory

logger = get_logger(__name__)
error_handler = get_error_handler()

class ConnectionStatus(Enum):
    """Connection status enumeration."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

@dataclass
class APIHealth:
    """API health status information."""
    status: ConnectionStatus
    response_time: float
    last_check: datetime
    error_count: int
    success_rate: float
    last_error: Optional[str] = None

class ConnectionMonitor:
    """Monitor API connections and health."""
    
    def __init__(self):
        self.config = get_config()
        self.health_data: Dict[str, APIHealth] = {}
        self.monitoring_active = False
        self._lock = threading.Lock()
        
        # Health check settings
        self.check_interval = self.config.error_handling.health_check_interval
        self.timeout = self.config.error_handling.connection_timeout
        
        # Initialize health data
        self._initialize_health_data()
    
    def _initialize_health_data(self) -> None:
        """Initialize health data for monitored services."""
        services = ["gemini_api", "internet", "system"]
        
        for service in services:
            self.health_data[service] = APIHealth(
                status=ConnectionStatus.UNKNOWN,
                response_time=0.0,
                last_check=datetime.now(),
                error_count=0,
                success_rate=1.0
            )
    
    def start_monitoring(self) -> None:
        """Start background health monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    self._check_all_services()
                    time.sleep(self.check_interval)
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                    time.sleep(30)  # Wait before retrying
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        logger.info("Connection monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        self.monitoring_active = False
        logger.info("Connection monitoring stopped")
    
    def _check_all_services(self) -> None:
        """Check health of all monitored services."""
        with self._lock:
            # Check internet connectivity
            self._check_internet_health()
            
            # Check Gemini API
            self._check_gemini_health()
            
            # Check system resources
            self._check_system_health()
    
    def _check_internet_health(self) -> None:
        """Check internet connectivity."""
        service = "internet"
        start_time = time.time()
        
        try:
            response = requests.get(
                "https://www.google.com",
                timeout=self.timeout,
                headers={'User-Agent': 'AI-Stock-Chart-Assistant/1.0'}
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                self._update_health_success(service, response_time)
            else:
                self._update_health_error(service, f"HTTP {response.status_code}")
                
        except Exception as e:
            self._update_health_error(service, str(e))
    
    def _check_gemini_health(self) -> None:
        """Check Gemini API health."""
        service = "gemini_api"
        start_time = time.time()
        
        try:
            # Simple API test (this would need actual implementation)
            # For now, we'll check if the API key is configured
            api_config = self.config.get_api_config()
            
            if not api_config.get("api_key"):
                self._update_health_error(service, "API key not configured")
                return
            
            # Simulate API health check
            response_time = time.time() - start_time
            self._update_health_success(service, response_time)
            
        except Exception as e:
            self._update_health_error(service, str(e))
    
    def _check_system_health(self) -> None:
        """Check system resource health."""
        service = "system"
        start_time = time.time()
        
        try:
            import psutil
            
            # Check memory usage
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            response_time = time.time() - start_time
            
            # Consider system healthy if memory < 90% and disk < 90%
            if memory.percent < 90 and disk.percent < 90:
                self._update_health_success(service, response_time)
            else:
                error_msg = f"High resource usage: Memory {memory.percent}%, Disk {disk.percent}%"
                self._update_health_error(service, error_msg)
                
        except ImportError:
            # psutil not available
            response_time = time.time() - start_time
            self._update_health_success(service, response_time)
        except Exception as e:
            self._update_health_error(service, str(e))
    
    def _update_health_success(self, service: str, response_time: float) -> None:
        """Update health data for successful check."""
        health = self.health_data[service]
        
        # Update status based on response time
        if response_time < 1.0:
            status = ConnectionStatus.CONNECTED
        elif response_time < 5.0:
            status = ConnectionStatus.DEGRADED
        else:
            status = ConnectionStatus.DEGRADED
        
        # Calculate success rate (simple moving average)
        total_checks = health.error_count + 1
        health.success_rate = (health.success_rate * (total_checks - 1) + 1.0) / total_checks
        
        # Update health data
        health.status = status
        health.response_time = response_time
        health.last_check = datetime.now()
        health.last_error = None
        
        logger.debug(f"{service} health check passed: {response_time:.2f}s")
    
    def _update_health_error(self, service: str, error_message: str) -> None:
        """Update health data for failed check."""
        health = self.health_data[service]
        
        health.status = ConnectionStatus.DISCONNECTED
        health.error_count += 1
        health.last_check = datetime.now()
        health.last_error = error_message
        
        # Update success rate
        total_checks = health.error_count + 1
        health.success_rate = health.success_rate * (total_checks - 1) / total_checks
        
        logger.warning(f"{service} health check failed: {error_message}")
    
    def get_health_status(self, service: Optional[str] = None) -> Dict[str, APIHealth]:
        """Get health status for services."""
        with self._lock:
            if service:
                return {service: self.health_data.get(service)}
            return self.health_data.copy()
    
    def is_service_healthy(self, service: str) -> bool:
        """Check if a specific service is healthy."""
        health = self.health_data.get(service)
        if not health:
            return False
        
        return health.status in [ConnectionStatus.CONNECTED, ConnectionStatus.DEGRADED]

class APIClient:
    """Robust API client with retry logic and monitoring."""
    
    def __init__(self):
        self.config = get_config()
        self.monitor = ConnectionMonitor()
        
        # Setup requests session with retry strategy
        self.session = self._create_session()
        
        # Rate limiting
        self.rate_limits = {
            "gemini": {"requests": 0, "window_start": time.time(), "max_per_minute": 60}
        }
    
    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy."""
        session = requests.Session()
        
        # Define retry strategy
        retry_strategy = Retry(
            total=self.config.error_handling.max_retry_attempts,
            backoff_factor=self.config.error_handling.retry_delay_base,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST"]
        )
        
        # Mount adapter with retry strategy
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers
        session.headers.update({
            'User-Agent': f'AI-Stock-Chart-Assistant/{self.config.version}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        return session
    
    def start_monitoring(self) -> None:
        """Start connection monitoring."""
        self.monitor.start_monitoring()
    
    def stop_monitoring(self) -> None:
        """Stop connection monitoring."""
        self.monitor.stop_monitoring()
    
    @handle_exception
    def make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Make HTTP request with error handling and monitoring.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            **kwargs: Additional request parameters
            
        Returns:
            Response object
        """
        # Check if internet is available
        if not self.monitor.is_service_healthy("internet"):
            raise ConnectionError("No internet connection available")
        
        # Set default timeout
        kwargs.setdefault('timeout', self.config.api.timeout_seconds)
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            
            logger.debug(f"API request successful: {method} {url}")
            return response
            
        except requests.exceptions.Timeout:
            error_msg = f"Request timeout: {url}"
            logger.error(error_msg)
            raise TimeoutError(error_msg)
            
        except requests.exceptions.ConnectionError:
            error_msg = f"Connection error: {url}"
            logger.error(error_msg)
            raise ConnectionError(error_msg)
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                error_msg = "Rate limit exceeded"
                logger.warning(error_msg)
                raise Exception(error_msg)
            else:
                error_msg = f"HTTP error {e.response.status_code}: {url}"
                logger.error(error_msg)
                raise
    
    def _check_rate_limit(self, service: str) -> None:
        """Check and enforce rate limits."""
        current_time = time.time()
        rate_limit = self.rate_limits.get(service, {})
        
        # Reset window if needed
        if current_time - rate_limit.get("window_start", 0) > 60:
            rate_limit["requests"] = 0
            rate_limit["window_start"] = current_time
        
        # Check if we're within limits
        max_requests = rate_limit.get("max_per_minute", 60)
        if rate_limit.get("requests", 0) >= max_requests:
            wait_time = 60 - (current_time - rate_limit.get("window_start", 0))
            if wait_time > 0:
                logger.warning(f"Rate limit reached for {service}, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                rate_limit["requests"] = 0
                rate_limit["window_start"] = time.time()
        
        # Increment request count
        rate_limit["requests"] = rate_limit.get("requests", 0) + 1
        self.rate_limits[service] = rate_limit
    
    @handle_exception
    def test_connectivity(self) -> Dict[str, bool]:
        """Test connectivity to various services."""
        results = {}
        
        # Test internet connectivity
        try:
            response = self.make_request("GET", "https://www.google.com")
            results["internet"] = response.status_code == 200
        except Exception:
            results["internet"] = False
        
        # Test Gemini API (basic check)
        try:
            api_config = self.config.get_api_config()
            results["gemini_api"] = bool(api_config.get("api_key"))
        except Exception:
            results["gemini_api"] = False
        
        return results
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get comprehensive connection status."""
        health_status = self.monitor.get_health_status()
        
        status = {
            "overall_healthy": all(
                health.status in [ConnectionStatus.CONNECTED, ConnectionStatus.DEGRADED]
                for health in health_status.values()
            ),
            "services": {}
        }
        
        for service, health in health_status.items():
            status["services"][service] = {
                "status": health.status.value,
                "response_time": health.response_time,
                "success_rate": health.success_rate,
                "last_check": health.last_check.isoformat(),
                "error_count": health.error_count,
                "last_error": health.last_error
            }
        
        return status
    
    async def async_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """
        Make asynchronous HTTP request.
        
        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional request parameters
            
        Returns:
            Response data as dictionary
        """
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, **kwargs) as response:
                    response.raise_for_status()
                    return await response.json()
                    
        except Exception as e:
            error_info = error_handler.handle_error(e, f"Async request: {method} {url}")
            logger.error(f"Async request failed: {error_info.user_message}")
            raise
    
    def batch_requests(self, requests_data: List[Dict[str, Any]]) -> List[Optional[requests.Response]]:
        """
        Execute multiple requests in batch.
        
        Args:
            requests_data: List of request dictionaries with 'method', 'url', and optional kwargs
            
        Returns:
            List of response objects (None for failed requests)
        """
        results = []
        
        for request_data in requests_data:
            try:
                method = request_data.pop("method")
                url = request_data.pop("url")
                
                response = self.make_request(method, url, **request_data)
                results.append(response)
                
            except Exception as e:
                error_info = error_handler.handle_error(e, f"Batch request: {request_data}")
                logger.error(f"Batch request failed: {error_info.user_message}")
                results.append(None)
        
        return results
    
    def close(self) -> None:
        """Clean up resources."""
        self.session.close()
        self.stop_monitoring()
        logger.info("API client closed")

class GeminiAPIClient:
    """Specialized client for Google Gemini API."""
    
    def __init__(self):
        self.config = get_config()
        self.api_config = self.config.get_api_config()
        self.client = APIClient()
        
        # Initialize Gemini
        self._initialize_gemini()
        
        # Rate limiting specific to Gemini
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests
    
    def _initialize_gemini(self) -> None:
        """Initialize Gemini API client."""
        try:
            if not self.api_config["api_key"]:
                raise ValueError("Gemini API key not configured")
            
            genai.configure(api_key=self.api_config["api_key"])
            self.model = genai.GenerativeModel(self.api_config["model"])
            
            logger.info("Gemini API client initialized")
            
        except Exception as e:
            error_info = error_handler.handle_error(e, "Gemini API initialization")
            logger.error(f"Failed to initialize Gemini API: {error_info.user_message}")
            raise
    
    @handle_exception
    def generate_content(self, prompt: str, image_data: Optional[bytes] = None) -> str:
        """
        Generate content using Gemini API.
        
        Args:
            prompt: Text prompt
            image_data: Optional image data for vision tasks
            
        Returns:
            Generated content as string
        """
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            time.sleep(wait_time)
        
        try:
            # Check service health
            if not self.client.monitor.is_service_healthy("gemini_api"):
                raise ConnectionError("Gemini API is not healthy")
            
            # Prepare content
            content = [prompt]
            
            if image_data:
                from PIL import Image
                import io
                image = Image.open(io.BytesIO(image_data))
                content.append(image)
            
            # Generate content
            response = self.model.generate_content(content)
            
            if not response.text:
                raise ValueError("Empty response from Gemini API")
            
            self.last_request_time = time.time()
            logger.debug("Gemini content generation successful")
            
            return response.text
            
        except Exception as e:
            # Handle specific Gemini errors
            error_msg = str(e).lower()
            
            if "quota" in error_msg:
                raise Exception("Gemini API quota exceeded")
            elif "rate" in error_msg:
                raise Exception("Gemini API rate limit exceeded")
            elif "authentication" in error_msg or "api key" in error_msg:
                raise Exception("Gemini API authentication failed")
            else:
                raise
    
    def test_connection(self) -> Tuple[bool, str]:
        """Test Gemini API connection."""
        try:
            # Simple test request
            test_prompt = "Respond with 'Connection test successful' if you can see this."
            response = self.generate_content(test_prompt)
            
            if "successful" in response.lower():
                return True, "Gemini API connection successful"
            else:
                return True, "Gemini API responding but unexpected response"
                
        except Exception as e:
            error_info = error_handler.handle_error(e, "Gemini API connection test")
            return False, error_info.user_message

# Global instances
api_client = APIClient()
gemini_client = GeminiAPIClient()

def get_api_client() -> APIClient:
    """Get the global API client instance."""
    return api_client

def get_gemini_client() -> GeminiAPIClient:
    """Get the global Gemini API client instance."""
    return gemini_client

if __name__ == "__main__":
    # API client test
    print("AI Stock Chart Assistant - API Client Test")
    print("=" * 50)
    
    # Test connectivity
    connectivity = api_client.test_connectivity()
    for service, status in connectivity.items():
        status_icon = "✓" if status else "✗"
        print(f"{status_icon} {service}: {'Connected' if status else 'Disconnected'}")
    
    # Test Gemini API
    success, message = gemini_client.test_connection()
    status_icon = "✓" if success else "✗"
    print(f"{status_icon} Gemini API: {message}")
    
    # Start monitoring
    api_client.start_monitoring()
    print("✓ Connection monitoring started") 