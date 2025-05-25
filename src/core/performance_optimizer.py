"""
Performance Optimizer for AI Stock Chart Assistant v2.0

Provides performance optimizations including:
- Image compression and optimization
- Memory management
- Parallel processing
- Resource monitoring
"""

import asyncio
import concurrent.futures
import logging
import os
import psutil
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
import joblib


@dataclass
class PerformanceStats:
    """Performance monitoring statistics"""
    images_processed: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hit_rate: float = 0.0
    parallel_tasks_executed: int = 0


class PerformanceOptimizer:
    """
    Performance optimization manager for image processing and resource management
    """
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 max_image_size: Tuple[int, int] = (2048, 2048),
                 compression_quality: int = 85,
                 memory_limit_mb: int = 1024):
        
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.max_image_size = max_image_size
        self.compression_quality = compression_quality
        self.memory_limit_mb = memory_limit_mb
        
        # Thread pool for CPU-bound tasks
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        )
        
        # Process pool for heavy computations
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=min(4, os.cpu_count() or 1)
        )
        
        # Statistics
        self.stats = PerformanceStats()
        self._stats_lock = threading.Lock()
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Resource monitoring
        self._start_monitoring()
    
    def optimize_image_for_api(self, image: Image.Image) -> Image.Image:
        """
        Optimize image for API consumption while preserving quality
        
        Args:
            image: PIL Image to optimize
            
        Returns:
            Optimized PIL Image
        """
        start_time = time.time()
        
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large
            if (image.width > self.max_image_size[0] or 
                image.height > self.max_image_size[1]):
                
                # Calculate new size maintaining aspect ratio
                ratio = min(
                    self.max_image_size[0] / image.width,
                    self.max_image_size[1] / image.height
                )
                new_size = (
                    int(image.width * ratio),
                    int(image.height * ratio)
                )
                
                # Use high-quality resampling
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                self.logger.debug(f"Resized image to {new_size}")
            
            # Auto-orient based on EXIF data
            image = ImageOps.exif_transpose(image)
            
            # Enhance contrast if needed
            image = self._enhance_image_quality(image)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(processing_time)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error optimizing image: {e}")
            return image
    
    def _enhance_image_quality(self, image: Image.Image) -> Image.Image:
        """Enhance image quality for better AI analysis"""
        try:
            # Convert to numpy array for OpenCV processing
            img_array = np.array(image)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Apply slight sharpening
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel * 0.1)
            
            # Blend original and enhanced
            result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
            
            return Image.fromarray(result)
            
        except Exception as e:
            self.logger.warning(f"Image enhancement failed, using original: {e}")
            return image
    
    async def process_images_parallel(self, 
                                    image_paths: List[str],
                                    processor_func: Callable,
                                    **kwargs) -> List[Any]:
        """
        Process multiple images in parallel
        
        Args:
            image_paths: List of image file paths
            processor_func: Function to process each image
            **kwargs: Additional arguments for processor function
            
        Returns:
            List of processing results
        """
        start_time = time.time()
        
        # Create tasks for parallel processing
        tasks = []
        for image_path in image_paths:
            task = asyncio.create_task(
                self._process_single_image(image_path, processor_func, **kwargs)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Error processing {image_paths[i]}: {result}")
            else:
                successful_results.append(result)
        
        # Update statistics
        processing_time = time.time() - start_time
        with self._stats_lock:
            self.stats.parallel_tasks_executed += len(image_paths)
        
        self.logger.info(f"Processed {len(successful_results)}/{len(image_paths)} images in {processing_time:.2f}s")
        
        return successful_results
    
    async def _process_single_image(self, 
                                  image_path: str, 
                                  processor_func: Callable,
                                  **kwargs) -> Any:
        """Process a single image asynchronously"""
        loop = asyncio.get_event_loop()
        
        # Run in thread pool to avoid blocking
        return await loop.run_in_executor(
            self.thread_pool,
            processor_func,
            image_path,
            **kwargs
        )
    
    def compress_image(self, 
                      image_path: str, 
                      output_path: Optional[str] = None,
                      quality: Optional[int] = None) -> str:
        """
        Compress image file while maintaining quality
        
        Args:
            image_path: Input image path
            output_path: Output path (default: overwrite input)
            quality: JPEG quality (default: use instance setting)
            
        Returns:
            Path to compressed image
        """
        if quality is None:
            quality = self.compression_quality
        
        if output_path is None:
            output_path = image_path
        
        try:
            with Image.open(image_path) as image:
                # Optimize image
                optimized = self.optimize_image_for_api(image)
                
                # Save with compression
                save_kwargs = {'quality': quality, 'optimize': True}
                
                if image_path.lower().endswith('.png'):
                    # For PNG, use compression level instead of quality
                    optimized.save(output_path, 'PNG', compress_level=6)
                else:
                    # For JPEG and others
                    optimized.save(output_path, 'JPEG', **save_kwargs)
                
                self.logger.debug(f"Compressed image: {image_path} -> {output_path}")
                return output_path
                
        except Exception as e:
            self.logger.error(f"Error compressing image {image_path}: {e}")
            return image_path
    
    def batch_compress_images(self, 
                            image_paths: List[str],
                            output_dir: Optional[str] = None,
                            quality: Optional[int] = None) -> List[str]:
        """
        Compress multiple images in parallel
        
        Args:
            image_paths: List of input image paths
            output_dir: Output directory (default: same as input)
            quality: JPEG quality
            
        Returns:
            List of output paths
        """
        if quality is None:
            quality = self.compression_quality
        
        # Prepare output paths
        output_paths = []
        for image_path in image_paths:
            if output_dir:
                filename = Path(image_path).name
                output_path = str(Path(output_dir) / filename)
            else:
                output_path = image_path
            output_paths.append(output_path)
        
        # Process in parallel using joblib
        results = joblib.Parallel(n_jobs=self.max_workers)(
            joblib.delayed(self.compress_image)(
                image_path, output_path, quality
            )
            for image_path, output_path in zip(image_paths, output_paths)
        )
        
        return results
    
    def monitor_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage is within limits"""
        memory_stats = self.monitor_memory_usage()
        return memory_stats['rss_mb'] < self.memory_limit_mb
    
    async def cleanup_memory(self):
        """Force garbage collection and memory cleanup"""
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Clear any cached data if memory is tight
        if not self.check_memory_limit():
            self.logger.warning("Memory usage high, forcing cleanup")
            
            # Additional cleanup can be added here
            # e.g., clearing image caches, temporary files, etc.
    
    def get_optimal_batch_size(self, 
                             item_size_mb: float,
                             max_memory_mb: Optional[float] = None) -> int:
        """
        Calculate optimal batch size based on available memory
        
        Args:
            item_size_mb: Average size of each item in MB
            max_memory_mb: Maximum memory to use (default: 50% of available)
            
        Returns:
            Optimal batch size
        """
        if max_memory_mb is None:
            available_mb = psutil.virtual_memory().available / 1024 / 1024
            max_memory_mb = available_mb * 0.5  # Use 50% of available memory
        
        if item_size_mb <= 0:
            return 1
        
        batch_size = max(1, int(max_memory_mb / item_size_mb))
        return min(batch_size, self.max_workers * 2)  # Don't exceed worker capacity
    
    def _start_monitoring(self):
        """Start background resource monitoring"""
        def monitor_loop():
            while True:
                try:
                    # Update memory and CPU stats
                    memory_stats = self.monitor_memory_usage()
                    cpu_percent = psutil.cpu_percent(interval=1)
                    
                    with self._stats_lock:
                        self.stats.memory_usage_mb = memory_stats['rss_mb']
                        self.stats.cpu_usage_percent = cpu_percent
                    
                    # Sleep for monitoring interval
                    time.sleep(30)  # Monitor every 30 seconds
                    
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(60)  # Wait longer on error
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def _update_stats(self, processing_time: float):
        """Update performance statistics"""
        with self._stats_lock:
            self.stats.images_processed += 1
            self.stats.total_processing_time += processing_time
            self.stats.average_processing_time = (
                self.stats.total_processing_time / self.stats.images_processed
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self._stats_lock:
            return {
                'images_processed': self.stats.images_processed,
                'total_processing_time': self.stats.total_processing_time,
                'average_processing_time': self.stats.average_processing_time,
                'memory_usage_mb': self.stats.memory_usage_mb,
                'cpu_usage_percent': self.stats.cpu_usage_percent,
                'parallel_tasks_executed': self.stats.parallel_tasks_executed,
                'max_workers': self.max_workers,
                'memory_limit_mb': self.memory_limit_mb
            }
    
    def reset_stats(self):
        """Reset performance statistics"""
        with self._stats_lock:
            self.stats = PerformanceStats()
    
    def close(self):
        """Cleanup resources"""
        try:
            self.thread_pool.shutdown(wait=True)
            self.process_pool.shutdown(wait=True)
            self.logger.info("Performance optimizer closed")
        except Exception as e:
            self.logger.error(f"Error closing performance optimizer: {e}")


class ImageProcessor:
    """
    Specialized image processing utilities
    """
    
    @staticmethod
    def extract_chart_region(image: Image.Image) -> Image.Image:
        """Extract the main chart region from a stock chart image"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest rectangular contour (likely the chart area)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Add some padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.width - x, w + 2 * padding)
                h = min(image.height - y, h + 2 * padding)
                
                # Crop the image
                cropped = image.crop((x, y, x + w, y + h))
                return cropped
            
            return image
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Chart region extraction failed: {e}")
            return image
    
    @staticmethod
    def enhance_chart_visibility(image: Image.Image) -> Image.Image:
        """Enhance chart visibility for better AI analysis"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Enhance contrast
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Reduce noise
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            return Image.fromarray(denoised)
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Chart enhancement failed: {e}")
            return image
    
    @staticmethod
    def detect_chart_type(image: Image.Image) -> str:
        """Detect the type of chart (candlestick, line, bar, etc.)"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Look for vertical lines (candlesticks)
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
            
            # Look for horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Count features
            vertical_features = cv2.countNonZero(vertical_lines)
            horizontal_features = cv2.countNonZero(horizontal_lines)
            
            # Simple heuristic for chart type detection
            if vertical_features > horizontal_features * 2:
                return "candlestick"
            elif horizontal_features > vertical_features:
                return "line"
            else:
                return "mixed"
                
        except Exception:
            return "unknown" 