#!/usr/bin/env python3
"""
AI Stock Chart Assistant - Enhanced Error Handling Version
Robust application with retry mechanisms, recovery, and graceful degradation
Author: Boworn Treesinsub
Version: 1.1 - Enhanced Error Handling
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import google.generativeai as genai
import os
import threading
import json
from typing import Optional, Dict
from datetime import datetime

# Import enhanced error handling
try:
    from enhanced_error_handler import (
        enhanced_exception_handler, retry_on_failure, SafeOperation,
        ConnectionManager, OfflineMode, recovery_manager, connection_manager,
        validate_system_requirements, safe_file_operation
    )
    ENHANCED_ERROR_HANDLING = True
except ImportError:
    ENHANCED_ERROR_HANDLING = False
    def enhanced_exception_handler(show_message=True, save_state=False):
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
    
    def retry_on_failure(max_retries=3, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class SafeOperation:
        def __init__(self, name, **kwargs):
            self.name = name
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

# Import configuration
try:
    from config import AppConfig
except ImportError:
    class AppConfig:
        APP_NAME = "AI Stock Chart Assistant"
        APP_VERSION = "1.1 Enhanced"
        WINDOW_SIZE = "1200x800"
        MIN_WINDOW_SIZE = (900, 700)
        SUPPORTED_FORMATS = [("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"), ("PNG files", "*.png"), ("JPEG files", "*.jpg *.jpeg"), ("All files", "*.*")]
        PREVIEW_SIZE = (300, 200)
        MAX_IMAGE_SIZE = 20 * 1024 * 1024
        GEMINI_MODEL = "gemini-1.5-flash"
        DEFAULT_QUESTION = "What insights can you provide about this stock chart?"
        ANALYSIS_PROMPT_TEMPLATE = """Analyze this stock chart and provide insights about: trends, patterns, support/resistance levels, indicators, and key observations. User question: {question}"""
        DISCLAIMER_TEXT = "Educational tool - not financial advice. Consult professionals before investing."
        
        @classmethod
        def get_api_key(cls):
            return os.getenv('GEMINI_API_KEY', '')
        
        @classmethod
        def validate_image_size(cls, file_path):
            try:
                return os.path.getsize(file_path) <= cls.MAX_IMAGE_SIZE
            except OSError:
                return False

# Configure appearance
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")

class EnhancedStockChartAnalyzer:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title(f"📈 {AppConfig.APP_NAME} v{AppConfig.APP_VERSION}")
        self.root.geometry(AppConfig.WINDOW_SIZE)
        self.root.minsize(*AppConfig.MIN_WINDOW_SIZE)
        
        # Initialize variables
        self.current_image_path = None
        self.current_image = None
        self.api_key = None
        self.model = None
        self.api_connected = False
        
        # Initialize application
        self.setup_api()
        self.create_gui()
    
    def setup_api(self):
        """Setup API with error handling"""
        try:
            self.api_key = AppConfig.get_api_key()
            if not self.api_key:
                self.prompt_for_api_key()
                return
            
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(AppConfig.GEMINI_MODEL)
            self.api_connected = True
            print("✅ API connected successfully")
        except Exception as e:
            self.api_connected = False
            messagebox.showwarning("API Setup", f"AI service unavailable: {str(e)}\n\nYou can still upload and preview images.")
    
    def prompt_for_api_key(self):
        """Prompt for API key"""
        dialog = ctk.CTkInputDialog(
            text="Enter your Google Gemini API Key:\n(Get free key at: https://makersuite.google.com)",
            title="🔑 API Key Required"
        )
        api_key = dialog.get_input()
        
        if api_key:
            self.api_key = api_key
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(AppConfig.GEMINI_MODEL)
                self.api_connected = True
                messagebox.showinfo("✅ Success!", "API key configured!")
            except Exception as e:
                messagebox.showerror("❌ API Key Error", f"Failed to validate API key: {str(e)}")
                self.api_connected = False
    
    def create_gui(self):
        """Create the GUI"""
        # Main container
        main_frame = ctk.CTkFrame(self.root, corner_radius=15, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=25, pady=25)
        
        # Status bar
        self.create_status_bar(main_frame)
        
        # Header
        header_frame = ctk.CTkFrame(main_frame, corner_radius=15, height=80)
        header_frame.pack(fill="x", pady=(10, 25))
        header_frame.pack_propagate(False)
        
        title_text = f"📈 {AppConfig.APP_NAME} v{AppConfig.APP_VERSION}"
        if ENHANCED_ERROR_HANDLING:
            title_text += " - Enhanced"
        
        welcome_label = ctk.CTkLabel(header_frame, text=title_text, font=ctk.CTkFont(size=22, weight="bold"), text_color="#2E8B57")
        welcome_label.pack(pady=(10, 5))
        
        subtitle_label = ctk.CTkLabel(header_frame, text="AI-powered stock chart analysis with enhanced reliability", font=ctk.CTkFont(size=13), text_color="#696969")
        subtitle_label.pack()
        
        # Main content
        content_frame = ctk.CTkFrame(main_frame, corner_radius=15, fg_color="transparent")
        content_frame.pack(fill="both", expand=True)
        
        # Left column
        left_column = ctk.CTkFrame(content_frame, corner_radius=15, width=400)
        left_column.pack(side="left", fill="both", expand=True, padx=(0, 15))
        
        self.create_upload_section(left_column)
        self.create_question_section(left_column)
        self.create_action_section(left_column)
        
        # Right column
        right_column = ctk.CTkFrame(content_frame, corner_radius=15)
        right_column.pack(side="right", fill="both", expand=True)
        
        self.create_preview_section(right_column)
        self.create_results_section(right_column)
        
        # Disclaimer
        disclaimer_frame = ctk.CTkFrame(main_frame, corner_radius=12, fg_color="#FFF5EE", height=50)
        disclaimer_frame.pack(fill="x", pady=(15, 0))
        disclaimer_frame.pack_propagate(False)
        
        disclaimer_label = ctk.CTkLabel(disclaimer_frame, text="💡 Educational tool - not financial advice. Enhanced error handling ensures reliable operation.", font=ctk.CTkFont(size=11), text_color="#CD853F")
        disclaimer_label.pack(expand=True)
    
    def create_status_bar(self, parent):
        """Create status bar"""
        status_frame = ctk.CTkFrame(parent, corner_radius=8, height=35, fg_color="#F0F8FF")
        status_frame.pack(fill="x", pady=(0, 10))
        status_frame.pack_propagate(False)
        
        # Connection status
        connection_text = "🟢 AI Connected" if self.api_connected else "🔴 AI Offline"
        self.connection_label = ctk.CTkLabel(status_frame, text=connection_text, font=ctk.CTkFont(size=11))
        self.connection_label.pack(side="left", padx=15, pady=8)
        
        # Enhanced indicator
        if ENHANCED_ERROR_HANDLING:
            enhanced_label = ctk.CTkLabel(status_frame, text="⚡ Enhanced Mode", font=ctk.CTkFont(size=11), text_color="#2E8B57")
            enhanced_label.pack(side="left", padx=15, pady=8)
        
        # Retry button
        self.retry_button = ctk.CTkButton(status_frame, text="🔄 Reconnect", command=self.retry_connection, height=25, width=100, font=ctk.CTkFont(size=10))
        self.retry_button.pack(side="right", padx=15, pady=5)
        
        if self.api_connected:
            self.retry_button.configure(state="disabled")
    
    def create_upload_section(self, parent):
        """Create upload section"""
        upload_section = ctk.CTkFrame(parent, corner_radius=12, fg_color="#F0F8FF")
        upload_section.pack(fill="x", padx=20, pady=20)
        
        upload_label = ctk.CTkLabel(upload_section, text="📁 Upload Chart Image", font=ctk.CTkFont(size=16, weight="bold"), text_color="#4682B4")
        upload_label.pack(pady=(15, 5))
        
        self.upload_button = ctk.CTkButton(upload_section, text="🖼️ Select Chart Image", command=self.upload_image, height=40, font=ctk.CTkFont(size=13, weight="bold"), corner_radius=10)
        self.upload_button.pack(pady=10)
        
        self.image_info_label = ctk.CTkLabel(upload_section, text="No image selected", font=ctk.CTkFont(size=11), text_color="#888888")
        self.image_info_label.pack(pady=(5, 15))
    
    def create_question_section(self, parent):
        """Create question section"""
        question_section = ctk.CTkFrame(parent, corner_radius=12, fg_color="#F5FFFA")
        question_section.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        question_label = ctk.CTkLabel(question_section, text="💬 Analysis Question", font=ctk.CTkFont(size=16, weight="bold"), text_color="#2E8B57")
        question_label.pack(pady=(15, 5))
        
        self.question_entry = ctk.CTkTextbox(question_section, height=70, corner_radius=8, font=ctk.CTkFont(size=12))
        self.question_entry.pack(fill="x", padx=15, pady=(5, 15))
        self.question_entry.insert("0.0", AppConfig.DEFAULT_QUESTION)
    
    def create_action_section(self, parent):
        """Create action buttons"""
        button_section = ctk.CTkFrame(parent, corner_radius=12, fg_color="#FFF8DC", height=100)
        button_section.pack(fill="x", padx=20, pady=(0, 20))
        button_section.pack_propagate(False)
        
        button_frame = ctk.CTkFrame(button_section, fg_color="transparent")
        button_frame.pack(expand=True, pady=15)
        
        self.analyze_button = ctk.CTkButton(button_frame, text="🔍 Analyze Chart", command=self.analyze_chart, height=40, width=150, font=ctk.CTkFont(size=13, weight="bold"), corner_radius=10)
        self.analyze_button.pack(side="left", padx=(0, 10))
        
        self.clear_button = ctk.CTkButton(button_frame, text="🗑️ Clear", command=self.clear_all, height=40, width=100, corner_radius=10)
        self.clear_button.pack(side="right")
        
        self.analyze_button.configure(state="disabled")
    
    def create_preview_section(self, parent):
        """Create preview section"""
        preview_section = ctk.CTkFrame(parent, corner_radius=12, height=250)
        preview_section.pack(fill="x", padx=20, pady=20)
        preview_section.pack_propagate(False)
        
        preview_title = ctk.CTkLabel(preview_section, text="🖼️ Chart Preview", font=ctk.CTkFont(size=14, weight="bold"))
        preview_title.pack(pady=(15, 10))
        
        preview_frame = ctk.CTkFrame(preview_section, corner_radius=8, height=180)
        preview_frame.pack(fill="x", padx=15, pady=(0, 15))
        preview_frame.pack_propagate(False)
        
        self.image_preview = ctk.CTkLabel(preview_frame, text="Chart preview will appear here", font=ctk.CTkFont(size=12), text_color="#888888")
        self.image_preview.pack(expand=True)
    
    def create_results_section(self, parent):
        """Create results section"""
        results_section = ctk.CTkFrame(parent, corner_radius=12)
        results_section.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        results_title = ctk.CTkLabel(results_section, text="🤖 AI Analysis Results", font=ctk.CTkFont(size=14, weight="bold"))
        results_title.pack(pady=(15, 10))
        
        self.results_textbox = ctk.CTkTextbox(results_section, corner_radius=8, font=ctk.CTkFont(size=12))
        self.results_textbox.pack(fill="both", expand=True, padx=15, pady=(0, 15))
    
    def upload_image(self):
        """Upload image with validation"""
        try:
            file_path = filedialog.askopenfilename(title="Select Stock Chart Image", filetypes=AppConfig.SUPPORTED_FORMATS)
            
            if not file_path:
                return
            
            if not AppConfig.validate_image_size(file_path):
                max_size_mb = AppConfig.MAX_IMAGE_SIZE // (1024 * 1024)
                messagebox.showerror("File Too Large", f"Image too large. Maximum size: {max_size_mb}MB")
                return
            
            image = Image.open(file_path)
            self.current_image_path = file_path
            self.current_image = image
            
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            self.image_info_label.configure(text=f"✅ {filename} ({file_size:.1f}MB)", text_color="#228B22")
            
            self.display_preview(image)
            self.analyze_button.configure(state="normal")
            self.results_textbox.delete("0.0", "end")
            self.results_textbox.insert("0.0", "✨ Image loaded successfully! Ready for analysis.")
            
        except Exception as e:
            messagebox.showerror("Upload Error", f"Failed to load image: {str(e)}")
    
    def display_preview(self, image):
        """Display image preview"""
        try:
            thumbnail = image.copy()
            thumbnail.thumbnail((280, 160), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(thumbnail)
            self.image_preview.configure(image=photo, text="")
            self.image_preview.image = photo
        except Exception as e:
            self.image_preview.configure(text="Preview unavailable")
    
    def analyze_chart(self):
        """Analyze chart with error handling"""
        if not self.current_image:
            messagebox.showwarning("No Image", "Please upload a chart image first.")
            return
        
        if not self.api_connected:
            if messagebox.askyesno("AI Offline", "AI service not connected. Try to reconnect?"):
                self.retry_connection()
                if not self.api_connected:
                    return
            else:
                return
        
        self.analyze_button.configure(state="disabled", text="🧠 Analyzing...")
        self.results_textbox.delete("0.0", "end")
        self.results_textbox.insert("0.0", "🤖 AI is analyzing your chart...")
        
        thread = threading.Thread(target=self._perform_analysis)
        thread.daemon = True
        thread.start()
    
    def _perform_analysis(self):
        """Perform analysis"""
        try:
            question = self.question_entry.get("0.0", "end").strip()
            if not question:
                question = AppConfig.DEFAULT_QUESTION
            
            prompt = AppConfig.ANALYSIS_PROMPT_TEMPLATE.format(question=question)
            response = self.model.generate_content([prompt, self.current_image])
            
            if hasattr(response, 'text') and response.text:
                result_text = f"🎯 **Analysis Results**\n{'='*40}\n\n{response.text}\n\n{'='*40}\n\n💡 Analysis completed successfully."
                self.root.after(0, self._update_success, result_text)
            else:
                raise Exception("Empty response from AI service")
                
        except Exception as e:
            error_msg = f"❌ Analysis failed: {str(e)}"
            self.root.after(0, self._update_error, error_msg)
    
    def _update_success(self, result):
        """Update on success"""
        self.results_textbox.delete("0.0", "end")
        self.results_textbox.insert("0.0", result)
        self.analyze_button.configure(state="normal", text="🔍 Analyze Chart")
    
    def _update_error(self, error):
        """Update on error"""
        self.results_textbox.delete("0.0", "end")
        self.results_textbox.insert("0.0", error)
        self.analyze_button.configure(state="normal", text="🔍 Retry Analysis")
    
    def retry_connection(self):
        """Retry API connection"""
        self.retry_button.configure(state="disabled", text="Connecting...")
        
        def _retry():
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(AppConfig.GEMINI_MODEL)
                self.api_connected = True
                self.root.after(0, self._connection_success)
            except Exception as e:
                self.root.after(0, self._connection_failed, str(e))
        
        thread = threading.Thread(target=_retry)
        thread.daemon = True
        thread.start()
    
    def _connection_success(self):
        """Handle connection success"""
        self.connection_label.configure(text="🟢 AI Connected")
        self.retry_button.configure(state="disabled", text="🔄 Reconnect")
        messagebox.showinfo("✅ Connected", "AI service connection restored!")
    
    def _connection_failed(self, error):
        """Handle connection failure"""
        self.retry_button.configure(state="normal", text="🔄 Reconnect")
        messagebox.showerror("❌ Connection Failed", f"Could not reconnect: {error}")
    
    def clear_all(self):
        """Clear all data"""
        if self.current_image:
            if messagebox.askyesno("Clear All", "This will clear your current work. Continue?"):
                self._do_clear()
        else:
            self._do_clear()
    
    def _do_clear(self):
        """Actually clear data"""
        self.current_image_path = None
        self.current_image = None
        
        self.image_info_label.configure(text="No image selected", text_color="#888888")
        self.image_preview.configure(image="", text="Chart preview will appear here")
        if hasattr(self.image_preview, 'image'):
            self.image_preview.image = None
        
        self.question_entry.delete("0.0", "end")
        self.question_entry.insert("0.0", AppConfig.DEFAULT_QUESTION)
        
        self.results_textbox.delete("0.0", "end")
        self.results_textbox.insert("0.0", "Ready for new analysis.")
        
        self.analyze_button.configure(state="disabled", text="🔍 Analyze Chart")
    
    def run(self):
        """Start the application"""
        try:
            print(f"🚀 Starting {AppConfig.APP_NAME} v{AppConfig.APP_VERSION}")
            if ENHANCED_ERROR_HANDLING:
                print("⚡ Enhanced error handling active")
            print("📋 Application ready!")
            self.root.mainloop()
        except KeyboardInterrupt:
            print("👋 Application closed by user")
        except Exception as e:
            print(f"💥 Application error: {e}")
            messagebox.showerror("Critical Error", f"Application error: {str(e)}")

def main():
    """Main entry point"""
    try:
        import sys
        
        if sys.version_info < (3, 8):
            messagebox.showerror("Version Error", "Python 3.8 or higher required")
            return
        
        print("🎯 Initializing Enhanced AI Stock Chart Assistant...")
        
        if ENHANCED_ERROR_HANDLING:
            print("✅ Enhanced error handling loaded")
        else:
            print("⚠️ Using basic error handling")
        
        app = EnhancedStockChartAnalyzer()
        app.run()
        
    except ImportError as e:
        error_msg = f"Missing dependencies: {str(e)}\nRun: pip install -r requirements.txt"
        print(f"❌ {error_msg}")
        messagebox.showerror("Dependencies Missing", error_msg)
    except Exception as e:
        error_msg = f"Failed to start: {str(e)}"
        print(f"💥 {error_msg}")
        messagebox.showerror("Startup Error", error_msg)

if __name__ == "__main__":
    main()
