#!/usr/bin/env python3
"""
AI Stock Chart Assistant
A desktop application for AI-powered stock chart analysis using Vision-Language Models
Author: Boworn Treesinsub
Version: 1.0 MVP
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import google.generativeai as genai
import os
import threading
from typing import Optional

# Import custom modules
try:
    from config import AppConfig
    from error_handler import handle_exceptions, ErrorContext, log_error, validate_api_response
except ImportError:
    # Fallback if modules aren't available
    class AppConfig:
        APP_NAME = "AI Stock Chart Assistant"
        APP_VERSION = "1.0 MVP"
        WINDOW_SIZE = "1000x700"
        MIN_WINDOW_SIZE = (800, 600)
        THEME_MODE = "dark"
        COLOR_THEME = "blue"
        SUPPORTED_FORMATS = [
            ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*")
        ]
        PREVIEW_SIZE = (180, 120)
        MAX_IMAGE_SIZE = 20 * 1024 * 1024
        GEMINI_MODEL = "gemini-1.5-flash"
        DEFAULT_QUESTION = "Analyze this stock chart and identify any notable patterns, trends, or technical indicators."
        ANALYSIS_PROMPT_TEMPLATE = """You are an expert technical analyst. Please analyze the stock chart image and provide insights about:

1. Overall trend direction (bullish, bearish, or sideways)
2. Key support and resistance levels visible
3. Any chart patterns (triangles, head and shoulders, double tops/bottoms, etc.)
4. Technical indicators if visible (moving averages, RSI, MACD, etc.)
5. Volume patterns if shown
6. Potential price targets or levels to watch

User's specific question: {question}

Please provide a clear, structured analysis that would be helpful for educational purposes. Remember that this is for informational use only and not financial advice."""
        DISCLAIMER_TEXT = ("⚠️ DISCLAIMER: This tool is for informational and educational purposes only. "
                          "The AI analysis should not be considered as financial advice. "
                          "Always consult with qualified financial advisors before making investment decisions.")
        
        @classmethod
        def get_api_key(cls):
            return os.getenv('GEMINI_API_KEY', '')
        
        @classmethod
        def validate_image_size(cls, file_path):
            try:
                return os.path.getsize(file_path) <= cls.MAX_IMAGE_SIZE
            except OSError:
                return False
    
    def handle_exceptions(show_message=True):
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if show_message:
                        messagebox.showerror("Error", str(e))
                    return None
            return wrapper
        return decorator
    
    class ErrorContext:
        def __init__(self, operation_name, show_errors=True):
            self.operation_name = operation_name
            self.show_errors = show_errors
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type and self.show_errors:
                messagebox.showerror("Error", f"{self.operation_name} failed: {str(exc_val)}")
            return False

# Configure CustomTkinter appearance
ctk.set_appearance_mode(AppConfig.THEME_MODE)
ctk.set_default_color_theme(AppConfig.COLOR_THEME)

class StockChartAnalyzer:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title(f"{AppConfig.APP_NAME} v{AppConfig.APP_VERSION}")
        self.root.geometry(AppConfig.WINDOW_SIZE)
        self.root.minsize(*AppConfig.MIN_WINDOW_SIZE)
        
        # Initialize variables
        self.current_image_path: Optional[str] = None
        self.current_image: Optional[Image.Image] = None
        self.api_key: Optional[str] = None
        self.model = None
        
        # Configure Gemini API
        self.setup_api()
        
        # Create GUI
        self.create_widgets()
        
    @handle_exceptions(show_message=True)
    def setup_api(self):
        """Setup Google Gemini API with improved error handling"""
        with ErrorContext("API Setup"):
            # Try to get API key from environment variable
            self.api_key = AppConfig.get_api_key()
            
            if not self.api_key:
                self.prompt_for_api_key()
            else:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(AppConfig.GEMINI_MODEL)
                print("✅ Gemini API configured successfully")
    
    def prompt_for_api_key(self):
        """Prompt user for API key if not found in environment"""
        dialog = ctk.CTkInputDialog(
            text="Please enter your Google Gemini API Key:",
            title="API Key Required"
        )
        api_key = dialog.get_input()
        
        if api_key:
            self.api_key = api_key
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(AppConfig.GEMINI_MODEL)
                messagebox.showinfo("Success", "API key configured successfully!")
            except Exception as e:
                messagebox.showerror("API Error", f"Invalid API key: {str(e)}")
                self.root.quit()
        else:
            messagebox.showerror("Error", "API key is required to use this application.")
            self.root.quit()
    
    def create_widgets(self):
        """Create and arrange GUI widgets"""
        # Main container with padding
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = ctk.CTkLabel(
            main_frame, 
            text="AI Stock Chart Assistant", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=(0, 20))
        
        # Top section - Image upload and preview
        top_frame = ctk.CTkFrame(main_frame)
        top_frame.pack(fill="x", pady=(0, 20))
        
        # Image upload section
        upload_frame = ctk.CTkFrame(top_frame)
        upload_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        upload_label = ctk.CTkLabel(upload_frame, text="Upload Stock Chart Image", 
                                   font=ctk.CTkFont(size=16, weight="bold"))
        upload_label.pack(pady=(10, 5))
        
        self.upload_button = ctk.CTkButton(
            upload_frame, 
            text="Browse & Upload Image", 
            command=self.upload_image,
            height=40
        )
        self.upload_button.pack(pady=10)
        
        self.image_info_label = ctk.CTkLabel(upload_frame, text="No image selected")
        self.image_info_label.pack(pady=5)
        
        # Image preview section
        preview_frame = ctk.CTkFrame(top_frame)
        preview_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        preview_label = ctk.CTkLabel(preview_frame, text="Image Preview", 
                                    font=ctk.CTkFont(size=16, weight="bold"))
        preview_label.pack(pady=(10, 5))
        
        self.image_preview = ctk.CTkLabel(preview_frame, text="No image to preview", 
                                         width=200, height=150)
        self.image_preview.pack(pady=10, padx=10)
        
        # Question input section
        question_frame = ctk.CTkFrame(main_frame)
        question_frame.pack(fill="x", pady=(0, 20))
        
        question_label = ctk.CTkLabel(question_frame, text="Optional: Ask a specific question about the chart", 
                                     font=ctk.CTkFont(size=16, weight="bold"))
        question_label.pack(pady=(10, 5))
        
        self.question_entry = ctk.CTkTextbox(question_frame, height=60)
        self.question_entry.pack(fill="x", padx=10, pady=5)
        self.question_entry.insert("0.0", AppConfig.DEFAULT_QUESTION)
        
        # Control buttons
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(fill="x", pady=(0, 20))
        
        self.analyze_button = ctk.CTkButton(
            button_frame, 
            text="🔍 Analyze Chart", 
            command=self.analyze_chart,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.analyze_button.pack(side="left", padx=(10, 5), pady=10)
        
        self.clear_button = ctk.CTkButton(
            button_frame, 
            text="🗑️ Clear All", 
            command=self.clear_all,
            height=40
        )
        self.clear_button.pack(side="left", padx=5, pady=10)
        
        # Results section
        results_frame = ctk.CTkFrame(main_frame)
        results_frame.pack(fill="both", expand=True)
        
        results_label = ctk.CTkLabel(results_frame, text="AI Analysis Results", 
                                    font=ctk.CTkFont(size=16, weight="bold"))
        results_label.pack(pady=(10, 5))
        
        self.results_textbox = ctk.CTkTextbox(results_frame, height=200)
        self.results_textbox.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Disclaimer
        disclaimer_label = ctk.CTkLabel(
            main_frame, 
            text=AppConfig.DISCLAIMER_TEXT,
            font=ctk.CTkFont(size=10),
            text_color="orange",
            wraplength=950
        )
        disclaimer_label.pack(pady=(10, 0))
        
        # Initially disable analyze button
        self.analyze_button.configure(state="disabled")
    
    @handle_exceptions(show_message=True)
    def upload_image(self):
        """Handle image upload with improved validation"""
        with ErrorContext("Image Upload"):
            file_path = filedialog.askopenfilename(
                title="Select Stock Chart Image",
                filetypes=AppConfig.SUPPORTED_FORMATS
            )
            
            if not file_path:
                return
            
            # Validate file size
            if not AppConfig.validate_image_size(file_path):
                max_size_mb = AppConfig.MAX_IMAGE_SIZE // (1024 * 1024)
                messagebox.showerror("Error", f"Image file is too large. Maximum size is {max_size_mb}MB.")
                return
            
            # Load and validate image
            image = Image.open(file_path)
            self.current_image_path = file_path
            self.current_image = image
            
            # Update image info
            filename = os.path.basename(file_path)
            size_info = f"{image.size[0]}x{image.size[1]}"
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            self.image_info_label.configure(
                text=f"Selected: {filename} ({size_info}) - {file_size:.1f}MB"
            )
            
            # Create and display thumbnail
            self.display_image_preview(image)
            
            # Enable analyze button
            self.analyze_button.configure(state="normal")
            
            # Clear previous results
            self.results_textbox.delete("0.0", "end")
            
            print(f"✅ Image loaded successfully: {filename}")
            
    @handle_exceptions(show_message=False)
    def display_image_preview(self, image: Image.Image):
        """Display image preview with error handling"""
        try:
            # Create thumbnail
            thumbnail = image.copy()
            thumbnail.thumbnail(AppConfig.PREVIEW_SIZE, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(thumbnail)
            
            # Update preview label
            self.image_preview.configure(image=photo, text="")
            self.image_preview.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Preview error: {e}")
            self.image_preview.configure(text="Preview unavailable")
    
    def analyze_chart(self):
        """Analyze the uploaded chart using Gemini API with enhanced error handling"""
        if not self.current_image:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return
        
        if not self.model:
            messagebox.showerror("Error", "AI service not configured. Please check your API key.")
            return
        
        # Disable button and show progress
        self.analyze_button.configure(state="disabled", text="🔄 Analyzing...")
        self.results_textbox.delete("0.0", "end")
        self.results_textbox.insert("0.0", "🤖 AI is analyzing your chart, please wait...")
        
        # Run analysis in separate thread to avoid UI blocking
        thread = threading.Thread(target=self._perform_analysis)
        thread.daemon = True
        thread.start()
    
    def _perform_analysis(self):
        """Perform the actual API call in a separate thread with comprehensive error handling"""
        try:
            with ErrorContext("Chart Analysis", show_errors=False):
                # Get user question or use default
                question = self.question_entry.get("0.0", "end").strip()
                if not question:
                    question = AppConfig.DEFAULT_QUESTION
                
                # Prepare the prompt using template
                prompt = AppConfig.ANALYSIS_PROMPT_TEMPLATE.format(question=question)
                
                print(f"🔍 Starting analysis with question: {question[:50]}...")
                
                # Call Gemini API
                response = self.model.generate_content([prompt, self.current_image])
                
                # Validate response
                if hasattr(response, 'text') and response.text:
                    result_text = response.text
                    print("✅ Analysis completed successfully")
                else:
                    result_text = "❌ No analysis result received. Please try again."
                    print("⚠️ Empty response from API")
                
                # Update UI in main thread
                self.root.after(0, self._update_results, result_text)
                
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            error_msg = self._get_user_friendly_error_message(e)
            self.root.after(0, self._update_results, f"❌ Analysis failed: {error_msg}")
    
    def _get_user_friendly_error_message(self, exception: Exception) -> str:
        """Convert technical errors to user-friendly messages"""
        error_str = str(exception).lower()
        
        if "quota" in error_str or "rate" in error_str:
            return "API rate limit exceeded. Please wait a moment and try again."
        elif "key" in error_str and "invalid" in error_str:
            return "Invalid API key. Please check your Google Gemini API key."
        elif "network" in error_str or "connection" in error_str:
            return "Network connection error. Please check your internet connection."
        elif "timeout" in error_str:
            return "Request timed out. Please try again."
        else:
            return f"Service error: {str(exception)}"
    
    def _update_results(self, result_text: str):
        """Update results in the main thread"""
        self.results_textbox.delete("0.0", "end")
        self.results_textbox.insert("0.0", result_text)
        
        # Re-enable button
        self.analyze_button.configure(state="normal", text="🔍 Analyze Chart")
    
    def clear_all(self):
        """Clear all inputs and results"""
        self.current_image_path = None
        self.current_image = None
        
        # Reset UI elements
        self.image_info_label.configure(text="No image selected")
        self.image_preview.configure(image="", text="No image to preview")
        self.image_preview.image = None
        
        self.question_entry.delete("0.0", "end")
        self.question_entry.insert("0.0", AppConfig.DEFAULT_QUESTION)
        
        self.results_textbox.delete("0.0", "end")
        
        # Disable analyze button
        self.analyze_button.configure(state="disabled")
        
        print("🧹 All data cleared")
    
    def run(self):
        """Start the application with error handling"""
        try:
            print(f"🚀 Starting {AppConfig.APP_NAME} v{AppConfig.APP_VERSION}")
            print("📋 Application ready for use!")
            self.root.mainloop()
        except KeyboardInterrupt:
            print("👋 Application closed by user")
        except Exception as e:
            print(f"💥 Application error: {e}")
            messagebox.showerror("Fatal Error", f"Application encountered an error: {str(e)}")

def main():
    """Main entry point with comprehensive error handling"""
    try:
        # Check Python version
        import sys
        if sys.version_info < (3, 8):
            messagebox.showerror("Version Error", "Python 3.8 or higher is required.")
            return
        
        print("🎯 Initializing AI Stock Chart Assistant...")
        app = StockChartAnalyzer()
        app.run()
        
    except ImportError as e:
        error_msg = f"Missing required dependency: {str(e)}\nPlease run: pip install -r requirements.txt"
        print(f"❌ {error_msg}")
        messagebox.showerror("Dependency Error", error_msg)
    except Exception as e:
        error_msg = f"Application failed to start: {str(e)}"
        print(f"💥 {error_msg}")
        messagebox.showerror("Fatal Error", error_msg)

if __name__ == "__main__":
    main()
