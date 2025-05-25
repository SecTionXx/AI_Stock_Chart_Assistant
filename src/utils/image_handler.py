"""
AI Stock Chart Assistant - Image Handler Module
Comprehensive image processing, validation, and manipulation utilities.
"""

import os
import hashlib
import mimetypes
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import numpy as np

from ..core.config import get_config, get_logger
from ..core.error_handler import get_error_handler, handle_exception

logger = get_logger(__name__)
error_handler = get_error_handler()

@dataclass
class ImageInfo:
    """Image metadata and information."""
    path: str
    filename: str
    size_bytes: int
    dimensions: Tuple[int, int]
    format: str
    mode: str
    has_transparency: bool
    created_date: datetime
    file_hash: str
    is_valid: bool
    error_message: Optional[str] = None

class ImageProcessor:
    """Advanced image processing for stock charts."""
    
    def __init__(self):
        self.config = get_config()
        self.supported_formats = self.config.ui.supported_formats
        self.max_file_size = self.config.ui.max_file_size_mb * 1024 * 1024
        
        # Processing settings
        self.max_dimension = 2048
        self.quality_settings = {
            "high": 95,
            "medium": 85,
            "low": 70
        }
    
    @handle_exception
    def validate_image(self, image_path: str) -> ImageInfo:
        """
        Comprehensive image validation and metadata extraction.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            ImageInfo object with validation results and metadata
        """
        path = Path(image_path)
        
        try:
            # Basic file checks
            if not path.exists():
                return ImageInfo(
                    path=str(path), filename=path.name, size_bytes=0,
                    dimensions=(0, 0), format="", mode="", has_transparency=False,
                    created_date=datetime.now(), file_hash="", is_valid=False,
                    error_message="File not found"
                )
            
            # File size check
            size_bytes = path.stat().st_size
            if size_bytes > self.max_file_size:
                size_mb = size_bytes / (1024 * 1024)
                return ImageInfo(
                    path=str(path), filename=path.name, size_bytes=size_bytes,
                    dimensions=(0, 0), format="", mode="", has_transparency=False,
                    created_date=datetime.fromtimestamp(path.stat().st_ctime),
                    file_hash="", is_valid=False,
                    error_message=f"File too large: {size_mb:.1f}MB (max: {self.config.ui.max_file_size_mb}MB)"
                )
            
            # Format check
            if path.suffix.lower() not in self.supported_formats:
                return ImageInfo(
                    path=str(path), filename=path.name, size_bytes=size_bytes,
                    dimensions=(0, 0), format=path.suffix, mode="", has_transparency=False,
                    created_date=datetime.fromtimestamp(path.stat().st_ctime),
                    file_hash="", is_valid=False,
                    error_message=f"Unsupported format: {path.suffix}"
                )
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(path)
            
            # Open and validate image
            with Image.open(path) as img:
                # Get image info
                dimensions = img.size
                format_name = img.format or path.suffix[1:].upper()
                mode = img.mode
                has_transparency = img.mode in ('RGBA', 'LA') or 'transparency' in img.info
                
                # Verify image integrity
                img.verify()
                
                # Additional validation
                if dimensions[0] < 50 or dimensions[1] < 50:
                    return ImageInfo(
                        path=str(path), filename=path.name, size_bytes=size_bytes,
                        dimensions=dimensions, format=format_name, mode=mode,
                        has_transparency=has_transparency,
                        created_date=datetime.fromtimestamp(path.stat().st_ctime),
                        file_hash=file_hash, is_valid=False,
                        error_message="Image too small (minimum 50x50 pixels)"
                    )
                
                return ImageInfo(
                    path=str(path), filename=path.name, size_bytes=size_bytes,
                    dimensions=dimensions, format=format_name, mode=mode,
                    has_transparency=has_transparency,
                    created_date=datetime.fromtimestamp(path.stat().st_ctime),
                    file_hash=file_hash, is_valid=True
                )
                
        except Exception as e:
            error_info = error_handler.handle_error(e, f"Image validation: {image_path}")
            return ImageInfo(
                path=str(path), filename=path.name, size_bytes=size_bytes if 'size_bytes' in locals() else 0,
                dimensions=(0, 0), format="", mode="", has_transparency=False,
                created_date=datetime.now(), file_hash="", is_valid=False,
                error_message=error_info.user_message
            )
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    @handle_exception
    def optimize_for_analysis(self, image_path: str, output_path: Optional[str] = None) -> str:
        """
        Optimize image for AI analysis.
        
        Args:
            image_path: Path to input image
            output_path: Optional output path (if None, creates optimized version)
            
        Returns:
            Path to optimized image
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    logger.info(f"Converted image from {img.mode} to RGB")
                
                # Resize if too large
                if max(img.size) > self.max_dimension:
                    ratio = self.max_dimension / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    logger.info(f"Resized image from {img.size} to {new_size}")
                
                # Enhance image for better analysis
                img = self._enhance_chart_image(img)
                
                # Determine output path
                if output_path is None:
                    input_path = Path(image_path)
                    output_path = input_path.parent / f"{input_path.stem}_optimized.jpg"
                
                # Save optimized image
                img.save(output_path, 'JPEG', quality=self.quality_settings["high"], optimize=True)
                
                logger.info(f"Image optimized and saved to: {output_path}")
                return str(output_path)
                
        except Exception as e:
            error_info = error_handler.handle_error(e, f"Image optimization: {image_path}")
            logger.error(f"Failed to optimize image: {error_info.user_message}")
            raise
    
    def _enhance_chart_image(self, img: Image.Image) -> Image.Image:
        """Enhance image specifically for chart analysis."""
        try:
            # Enhance contrast for better readability
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.1)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.05)
            
            # Slight brightness adjustment if needed
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.02)
            
            return img
            
        except Exception as e:
            logger.warning(f"Image enhancement failed, using original: {e}")
            return img
    
    @handle_exception
    def create_thumbnail(self, image_path: str, size: Tuple[int, int] = (200, 200)) -> str:
        """
        Create thumbnail for image preview.
        
        Args:
            image_path: Path to input image
            size: Thumbnail size (width, height)
            
        Returns:
            Path to thumbnail image
        """
        try:
            input_path = Path(image_path)
            thumbnail_path = input_path.parent / f"{input_path.stem}_thumb.jpg"
            
            with Image.open(image_path) as img:
                # Create thumbnail maintaining aspect ratio
                img.thumbnail(size, Image.Resampling.LANCZOS)
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save thumbnail
                img.save(thumbnail_path, 'JPEG', quality=self.quality_settings["medium"])
                
                logger.debug(f"Thumbnail created: {thumbnail_path}")
                return str(thumbnail_path)
                
        except Exception as e:
            error_info = error_handler.handle_error(e, f"Thumbnail creation: {image_path}")
            logger.error(f"Failed to create thumbnail: {error_info.user_message}")
            raise
    
    @handle_exception
    def detect_chart_features(self, image_path: str) -> Dict[str, Any]:
        """
        Detect chart features using computer vision.
        
        Args:
            image_path: Path to chart image
            
        Returns:
            Dictionary with detected features
        """
        try:
            # Load image with OpenCV
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not load image with OpenCV")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            features = {
                "has_grid": self._detect_grid_lines(gray),
                "has_candlesticks": self._detect_candlesticks(gray),
                "has_line_chart": self._detect_line_chart(gray),
                "text_regions": self._detect_text_regions(gray),
                "chart_area": self._detect_chart_area(gray),
                "color_analysis": self._analyze_colors(img)
            }
            
            logger.debug(f"Chart features detected: {features}")
            return features
            
        except Exception as e:
            error_info = error_handler.handle_error(e, f"Feature detection: {image_path}")
            logger.warning(f"Feature detection failed: {error_info.user_message}")
            return {}
    
    def _detect_grid_lines(self, gray_img: np.ndarray) -> bool:
        """Detect grid lines in chart."""
        try:
            # Use Hough line detection
            edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            return lines is not None and len(lines) > 10
            
        except Exception:
            return False
    
    def _detect_candlesticks(self, gray_img: np.ndarray) -> bool:
        """Detect candlestick patterns."""
        try:
            # Look for rectangular patterns typical of candlesticks
            contours, _ = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            rectangular_contours = 0
            for contour in contours:
                # Approximate contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's roughly rectangular
                if len(approx) == 4:
                    rectangular_contours += 1
            
            return rectangular_contours > 5
            
        except Exception:
            return False
    
    def _detect_line_chart(self, gray_img: np.ndarray) -> bool:
        """Detect line chart patterns."""
        try:
            # Use edge detection to find continuous lines
            edges = cv2.Canny(gray_img, 50, 150)
            
            # Count edge pixels
            edge_pixels = np.sum(edges > 0)
            total_pixels = gray_img.shape[0] * gray_img.shape[1]
            edge_ratio = edge_pixels / total_pixels
            
            # Line charts typically have more continuous edges
            return edge_ratio > 0.02
            
        except Exception:
            return False
    
    def _detect_text_regions(self, gray_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect text regions in the image."""
        try:
            # Use morphological operations to find text regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            
            # Apply morphological operations
            morph = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size (typical text characteristics)
                if 10 < w < 200 and 5 < h < 50:
                    text_regions.append((x, y, w, h))
            
            return text_regions[:10]  # Return top 10 text regions
            
        except Exception:
            return []
    
    def _detect_chart_area(self, gray_img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect the main chart area."""
        try:
            # Find the largest rectangular region (likely the chart area)
            contours, _ = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            return (x, y, w, h)
            
        except Exception:
            return None
    
    def _analyze_colors(self, img: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution in the image."""
        try:
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Calculate color statistics
            mean_color = np.mean(img_rgb, axis=(0, 1))
            
            # Detect dominant colors (simplified)
            pixels = img_rgb.reshape(-1, 3)
            
            # Check for common chart colors
            has_red = np.any(pixels[:, 0] > 200) and np.any(pixels[:, 1] < 100) and np.any(pixels[:, 2] < 100)
            has_green = np.any(pixels[:, 1] > 200) and np.any(pixels[:, 0] < 100) and np.any(pixels[:, 2] < 100)
            
            return {
                "mean_color": mean_color.tolist(),
                "has_red": bool(has_red),
                "has_green": bool(has_green),
                "brightness": float(np.mean(mean_color))
            }
            
        except Exception:
            return {}
    
    @handle_exception
    def batch_process(self, image_paths: List[str], operation: str = "optimize") -> List[str]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of image paths to process
            operation: Operation to perform ("optimize", "thumbnail", "validate")
            
        Returns:
            List of output paths or validation results
        """
        results = []
        
        for image_path in image_paths:
            try:
                if operation == "optimize":
                    result = self.optimize_for_analysis(image_path)
                elif operation == "thumbnail":
                    result = self.create_thumbnail(image_path)
                elif operation == "validate":
                    result = self.validate_image(image_path)
                else:
                    raise ValueError(f"Unknown operation: {operation}")
                
                results.append(result)
                logger.debug(f"Processed {image_path}: {operation}")
                
            except Exception as e:
                error_info = error_handler.handle_error(e, f"Batch processing: {image_path}")
                logger.error(f"Failed to process {image_path}: {error_info.user_message}")
                results.append(None)
        
        return results
    
    def cleanup_temp_files(self, directory: str, pattern: str = "*_optimized.*") -> int:
        """
        Clean up temporary files.
        
        Args:
            directory: Directory to clean
            pattern: File pattern to match
            
        Returns:
            Number of files cleaned up
        """
        try:
            path = Path(directory)
            if not path.exists():
                return 0
            
            files_removed = 0
            for file_path in path.glob(pattern):
                try:
                    file_path.unlink()
                    files_removed += 1
                    logger.debug(f"Removed temp file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove {file_path}: {e}")
            
            logger.info(f"Cleaned up {files_removed} temporary files")
            return files_removed
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 0

# Global image processor instance
image_processor = ImageProcessor()

def get_image_processor() -> ImageProcessor:
    """Get the global image processor instance."""
    return image_processor

# Convenience functions
def validate_image(image_path: str) -> ImageInfo:
    """Validate an image file."""
    return image_processor.validate_image(image_path)

def optimize_image(image_path: str, output_path: Optional[str] = None) -> str:
    """Optimize an image for analysis."""
    return image_processor.optimize_for_analysis(image_path, output_path)

def create_thumbnail(image_path: str, size: Tuple[int, int] = (200, 200)) -> str:
    """Create a thumbnail of an image."""
    return image_processor.create_thumbnail(image_path, size)

if __name__ == "__main__":
    # Image processor test
    print("AI Stock Chart Assistant - Image Processor Test")
    print("=" * 50)
    
    processor = get_image_processor()
    print(f"✓ Image processor initialized")
    print(f"✓ Supported formats: {processor.supported_formats}")
    print(f"✓ Max file size: {processor.max_file_size / (1024*1024):.1f}MB")
    print(f"✓ Max dimension: {processor.max_dimension}px") 