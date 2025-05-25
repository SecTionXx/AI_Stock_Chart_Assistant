#!/usr/bin/env python3
"""
AI Stock Chart Assistant - Friendly Version
A user-friendly desktop application for AI-powered stock chart analysis
Author: Boworn Treesinsub
Version: 1.0 MVP - Friendly GUI
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
        WINDOW_SIZE = "1200x800"
        MIN_WINDOW_SIZE = (900, 700)
        THEME_MODE = "light"
        COLOR_THEME = "green"
        SUPPORTED_FORMATS = [
            ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*")
        ]
        PREVIEW_SIZE = (300, 200)
        MAX_IMAGE_SIZE = 20 * 1024 * 1024
        GEMINI_MODEL = "gemini-1.5-flash"
        DEFAULT_QUESTION = "What insights can you provide about this stock chart?"
        ANALYSIS_PROMPT_TEMPLATE = """You are a friendly and knowledgeable technical analyst helping someone understand their stock chart. Please analyze the image and provide insights in a clear, easy-to-understand way:

1. **Overall Trend**: Is the stock generally going up, down, or sideways?
2. **Key Levels**: What are the important support and resistance price levels?
3. **Chart Patterns**: Are there any recognizable patterns (like triangles, head and shoulders, etc.)?
4. **Technical Indicators**: What do any visible indicators suggest?
5. **Volume**: What does the trading volume tell us?
6. **What to Watch**: What price levels or signals should be monitored?

User's question: {question}

Please explain everything in simple terms that anyone can understand. Remember this is for educational purposes only, not financial advice."""
        DISCLAIMER_TEXT = ("💡 Educational Tool: This analysis is for learning purposes only and should not be considered financial advice. Always consult with qualified financial advisors before making investment decisions.")
        
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
                        messagebox.showerror("Oops!", f"Something went wrong: {str(e)}")
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
                messagebox.showerror("Oops!", f"{self.operation_name} encountered an issue: {str(exc_val)}")
            return False

# Configure CustomTkinter for friendly appearance
ctk.set_appearance_mode("light")  # Light mode is more friendly
ctk.set_default_color_theme("green")  # Green for financial/growth

class FriendlyStockChartAnalyzer:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title(f"📈 {AppConfig.APP_NAME} - Your AI Trading Assistant")
        self.root.geometry("1200x800")
        self.root.minsize(900, 700)
        
        # Initialize variables
        self.current_image_path: Optional[str] = None
        self.current_image: Optional[Image.Image] = None
        self.api_key: Optional[str] = None
        self.model = None
        
        # Configure Gemini API
        self.setup_api()
        
        # Create friendly GUI
        self.create_friendly_widgets()
        
    @handle_exceptions(show_message=True)
    def setup_api(self):
        """Setup Google Gemini API with friendly messages"""
        with ErrorContext("Setting up AI service"):
            self.api_key = AppConfig.get_api_key()
            
            if not self.api_key:
                self.prompt_for_api_key()
            else:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(AppConfig.GEMINI_MODEL)
                print("🤖 AI assistant ready!")
    
    def prompt_for_api_key(self):
        """Friendly API key prompt"""
        dialog = ctk.CTkInputDialog(
            text="To get started, please enter your Google Gemini API Key.\n(Don't have one? Get it free at https://makersuite.google.com)",
            title="🔑 Connect to AI Service"
        )
        api_key = dialog.get_input()
        
        if api_key:
            self.api_key = api_key
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(AppConfig.GEMINI_MODEL)
                messagebox.showinfo("🎉 Success!", "Great! Your AI assistant is now ready to help analyze charts.")
            except Exception as e:
                messagebox.showerror("Connection Issue", f"Couldn't connect to the AI service. Please check your API key.\n\nError: {str(e)}")
                self.root.quit()
        else:
            messagebox.showinfo("Setup Required", "You'll need an API key to use the AI features. You can get one free at https://makersuite.google.com")
            self.root.quit()
    
    def create_friendly_widgets(self):
        """Create a friendly, approachable GUI"""
        # Main container with softer padding
        main_frame = ctk.CTkFrame(self.root, corner_radius=15, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=25, pady=25)
        
        # Friendly welcome header
        header_frame = ctk.CTkFrame(main_frame, corner_radius=15, height=100)
        header_frame.pack(fill="x", pady=(0, 25))
        header_frame.pack_propagate(False)
        
        welcome_label = ctk.CTkLabel(
            header_frame, 
            text="📈 Welcome to Your AI Stock Chart Assistant! 🤖", 
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#2E8B57"
        )
        welcome_label.pack(pady=15)
        
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Upload a stock chart and get instant AI-powered insights to help you understand the patterns and trends",
            font=ctk.CTkFont(size=16),
            text_color="#696969"
        )
        subtitle_label.pack(pady=(0, 15))
        
        # Content area with two columns
        content_frame = ctk.CTkFrame(main_frame, corner_radius=15, fg_color="transparent")
        content_frame.pack(fill="both", expand=True)
        
        # Left column - Upload and controls
        left_column = ctk.CTkFrame(content_frame, corner_radius=15, width=400)
        left_column.pack(side="left", fill="both", expand=True, padx=(0, 15))
        left_column.pack_propagate(False)
        
        # Upload section with friendly styling
        upload_section = ctk.CTkFrame(left_column, corner_radius=12, fg_color="#F0F8FF")
        upload_section.pack(fill="x", padx=20, pady=20)
        
        upload_icon_label = ctk.CTkLabel(
            upload_section,
            text="📁 Step 1: Choose Your Chart",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#4682B4"
        )
        upload_icon_label.pack(pady=(15, 5))
        
        upload_desc = ctk.CTkLabel(
            upload_section,
            text="Select a stock chart image from your computer\n(PNG, JPG, GIF supported)",
            font=ctk.CTkFont(size=12),
            text_color="#696969"
        )
        upload_desc.pack(pady=(0, 10))
        
        self.upload_button = ctk.CTkButton(
            upload_section,
            text="🖼️ Browse & Upload Chart",
            command=self.upload_image,
            height=45,
            font=ctk.CTkFont(size=14, weight="bold"),
            corner_radius=10,
            fg_color="#32CD32",
            hover_color="#228B22"
        )
        self.upload_button.pack(pady=10)
        
        self.image_info_label = ctk.CTkLabel(
            upload_section,
            text="No chart selected yet",
            font=ctk.CTkFont(size=11),
            text_color="#888888"
        )
        self.image_info_label.pack(pady=(5, 15))
        
        # Question section
        question_section = ctk.CTkFrame(left_column, corner_radius=12, fg_color="#F5FFFA")
        question_section.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        question_icon_label = ctk.CTkLabel(
            question_section,
            text="💬 Step 2: What Would You Like to Know?",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#2E8B57"
        )
        question_icon_label.pack(pady=(15, 5))
        
        question_desc = ctk.CTkLabel(
            question_section,
            text="Ask specific questions or use the default analysis",
            font=ctk.CTkFont(size=12),
            text_color="#696969"
        )
        question_desc.pack(pady=(0, 10))
        
        self.question_entry = ctk.CTkTextbox(
            question_section,
            height=80,
            corner_radius=8,
            font=ctk.CTkFont(size=12),
            border_width=2,
            border_color="#90EE90"
        )
        self.question_entry.pack(fill="x", padx=15, pady=5)
        self.question_entry.insert("0.0", AppConfig.DEFAULT_QUESTION)
        
        # Action buttons
        button_section = ctk.CTkFrame(left_column, corner_radius=12, fg_color="#FFF8DC", height=120)
        button_section.pack(fill="x", padx=20, pady=(0, 20))
        button_section.pack_propagate(False)
        
        action_label = ctk.CTkLabel(
            button_section,
            text="🚀 Step 3: Get Your Analysis",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#DAA520"
        )
        action_label.pack(pady=(15, 10))
        
        button_frame = ctk.CTkFrame(button_section, fg_color="transparent")
        button_frame.pack(expand=True)
        
        self.analyze_button = ctk.CTkButton(
            button_frame,
            text="🔍 Analyze My Chart",
            command=self.analyze_chart,
            height=45,
            width=180,
            font=ctk.CTkFont(size=14, weight="bold"),
            corner_radius=10,
            fg_color="#FF6347",
            hover_color="#FF4500"
        )
        self.analyze_button.pack(side="left", padx=(0, 10))
        
        self.clear_button = ctk.CTkButton(
            button_frame,
            text="🗑️ Start Over",
            command=self.clear_all,
            height=45,
            width=120,
            font=ctk.CTkFont(size=12),
            corner_radius=10,
            fg_color="#708090",
            hover_color="#556B2F"
        )
        self.clear_button.pack(side="right")
        
        # Right column - Preview and results
        right_column = ctk.CTkFrame(content_frame, corner_radius=15)
        right_column.pack(side="right", fill="both", expand=True)
        
        # Image preview section
        preview_section = ctk.CTkFrame(right_column, corner_radius=12, height=280)
        preview_section.pack(fill="x", padx=20, pady=20)
        preview_section.pack_propagate(False)
        
        preview_title = ctk.CTkLabel(
            preview_section,
            text="🖼️ Your Chart Preview",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#4682B4"
        )
        preview_title.pack(pady=(15, 10))
        
        # Preview frame with border
        preview_frame = ctk.CTkFrame(preview_section, corner_radius=8, fg_color="#F8F8FF", height=200)
        preview_frame.pack(fill="x", padx=15, pady=(0, 15))
        preview_frame.pack_propagate(False)
        
        self.image_preview = ctk.CTkLabel(
            preview_frame,
            text="📊 Chart preview will appear here",
            font=ctk.CTkFont(size=14),
            text_color="#888888"
        )
        self.image_preview.pack(expand=True)
        
        # Results section
        results_section = ctk.CTkFrame(right_column, corner_radius=12)
        results_section.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        results_title = ctk.CTkLabel(
            results_section,
            text="🤖 AI Analysis Results",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#2E8B57"
        )
        results_title.pack(pady=(15, 10))
        
        self.results_textbox = ctk.CTkTextbox(
            results_section,
            corner_radius=8,
            font=ctk.CTkFont(size=13),
            border_width=2,
            border_color="#90EE90"
        )
        self.results_textbox.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        # Friendly disclaimer at bottom
        disclaimer_frame = ctk.CTkFrame(main_frame, corner_radius=12, fg_color="#FFF5EE", height=60)
        disclaimer_frame.pack(fill="x", pady=(15, 0))
        disclaimer_frame.pack_propagate(False)
        
        disclaimer_label = ctk.CTkLabel(
            disclaimer_frame,
            text=AppConfig.DISCLAIMER_TEXT,
            font=ctk.CTkFont(size=11),
            text_color="#CD853F",
            wraplength=1100
        )
        disclaimer_label.pack(expand=True)
        
        # Initially disable analyze button
        self.analyze_button.configure(state="disabled")
        
    @handle_exceptions(show_message=True)
    def upload_image(self):
        """Handle image upload with friendly feedback"""
        with ErrorContext("Uploading your chart"):
            file_path = filedialog.askopenfilename(
                title="Select Your Stock Chart Image",
                filetypes=AppConfig.SUPPORTED_FORMATS
            )
            
            if not file_path:
                return
            
            # Validate file size
            if not AppConfig.validate_image_size(file_path):
                max_size_mb = AppConfig.MAX_IMAGE_SIZE // (1024 * 1024)
                messagebox.showwarning("File Too Large", f"Your image is too big! Please choose an image smaller than {max_size_mb}MB.")
                return
            
            # Load and validate image
            image = Image.open(file_path)
            self.current_image_path = file_path
            self.current_image = image
            
            # Update image info with friendly message
            filename = os.path.basename(file_path)
            size_info = f"{image.size[0]}×{image.size[1]}"
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            self.image_info_label.configure(
                text=f"✅ Ready to analyze: {filename}\n📏 Size: {size_info} pixels • 📁 {file_size:.1f}MB",
                text_color="#228B22"
            )
            
            # Create and display thumbnail
            self.display_image_preview(image)
            
            # Enable analyze button with encouraging message
            self.analyze_button.configure(state="normal")
            
            # Clear previous results
            self.results_textbox.delete("0.0", "end")
            self.results_textbox.insert("0.0", "✨ Chart uploaded successfully! Click 'Analyze My Chart' to get your AI insights.")
            
            print(f"✅ Image loaded successfully: {filename}")
            
    @handle_exceptions(show_message=False)
    def display_image_preview(self, image: Image.Image):
        """Display image preview with friendly styling"""
        try:
            # Create larger, friendlier thumbnail
            thumbnail = image.copy()
            thumbnail.thumbnail((350, 250), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(thumbnail)
            
            # Update preview label
            self.image_preview.configure(image=photo, text="")
            self.image_preview.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Preview error: {e}")
            self.image_preview.configure(text="📊 Chart preview unavailable\n(Don't worry, analysis will still work!)")
    
    def analyze_chart(self):
        """Analyze chart with encouraging feedback"""
        if not self.current_image:
            messagebox.showinfo("Hold On!", "Please upload a chart image first. 📊")
            return
        
        if not self.model:
            messagebox.showerror("AI Service Issue", "The AI service isn't connected. Please check your API key.")
            return
        
        # Disable button and show encouraging progress
        self.analyze_button.configure(state="disabled", text="🧠 AI is thinking...")
        self.results_textbox.delete("0.0", "end")
        self.results_textbox.insert("0.0", "🤖 Your AI assistant is carefully analyzing your chart...\n\n⏰ This usually takes 10-30 seconds. Please wait!")
        
        # Run analysis in separate thread
        thread = threading.Thread(target=self._perform_friendly_analysis)
        thread.daemon = True
        thread.start()
    
    def _perform_friendly_analysis(self):
        """Perform analysis with friendly error handling"""
        try:
            with ErrorContext("Getting your chart analysis", show_errors=False):
                # Get user question
                question = self.question_entry.get("0.0", "end").strip()
                if not question:
                    question = AppConfig.DEFAULT_QUESTION
                
                # Prepare friendly prompt
                prompt = AppConfig.ANALYSIS_PROMPT_TEMPLATE.format(question=question)
                
                print(f"🔍 Starting friendly analysis...")
                
                # Call Gemini API
                response = self.model.generate_content([prompt, self.current_image])
                
                # Validate and format response
                if hasattr(response, 'text') and response.text:
                    result_text = f"🎯 **Your Chart Analysis Results**\n{'='*50}\n\n{response.text}\n\n{'='*50}\n\n💡 **Remember**: This analysis is for educational purposes to help you understand chart patterns. Always do your own research and consider consulting with financial professionals for investment decisions."
                    print("✅ Analysis completed successfully")
                else:
                    result_text = "🤔 Hmm, the AI didn't provide a response. This sometimes happens. Please try again!"
                    print("⚠️ Empty response from AI")
                
                # Update UI in main thread
                self.root.after(0, self._update_friendly_results, result_text)
                
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            error_msg = self._get_friendly_error_message(e)
            self.root.after(0, self._update_friendly_results, f"😕 **Oops! Something went wrong:**\n\n{error_msg}\n\n🔄 **What you can try:**\n• Check your internet connection\n• Make sure your API key is working\n• Try again in a few moments\n• Try a different chart image")
    
    def _get_friendly_error_message(self, exception: Exception) -> str:
        """Convert technical errors to friendly messages"""
        error_str = str(exception).lower()
        
        if "quota" in error_str or "rate" in error_str:
            return "You've reached the API usage limit. Please wait a few minutes and try again. 🕐"
        elif "key" in error_str and "invalid" in error_str:
            return "There's an issue with your API key. Please check it and try again. 🔑"
        elif "network" in error_str or "connection" in error_str:
            return "Can't connect to the AI service. Please check your internet connection. 🌐"
        elif "timeout" in error_str:
            return "The request took too long. Please try again. ⏱️"
        else:
            return f"Technical error: {str(exception)} 🔧"
    
    def _update_friendly_results(self, result_text: str):
        """Update results with friendly formatting"""
        self.results_textbox.delete("0.0", "end")
        self.results_textbox.insert("0.0", result_text)
        
        # Re-enable button with friendly text
        self.analyze_button.configure(state="normal", text="🔍 Analyze Another Chart")
    
    def clear_all(self):
        """Clear everything with friendly confirmation"""
        if self.current_image:
            if messagebox.askyesno("Start Over?", "This will clear your current chart and analysis. Are you sure?"):
                self._do_clear_all()
        else:
            self._do_clear_all()
    
    def _do_clear_all(self):
        """Actually clear all data"""
        self.current_image_path = None
        self.current_image = None
        
        # Reset UI elements with friendly messages
        self.image_info_label.configure(text="No chart selected yet", text_color="#888888")
        self.image_preview.configure(image="", text="📊 Chart preview will appear here")
        self.image_preview.image = None
        
        self.question_entry.delete("0.0", "end")
        self.question_entry.insert("0.0", AppConfig.DEFAULT_QUESTION)
        
        self.results_textbox.delete("0.0", "end")
        self.results_textbox.insert("0.0", "👋 Ready for a new chart analysis! Upload an image to get started.")
        
        # Reset analyze button
        self.analyze_button.configure(state="disabled", text="🔍 Analyze My Chart")
        
        print("🧹 All cleared - ready for new analysis!")
    
    def run(self):
        """Start the friendly application"""
        try:
            print(f"🚀 Starting {AppConfig.APP_NAME} - Friendly Edition")
            print("🎉 Welcome! Your AI stock chart assistant is ready to help!")
            self.root.mainloop()
        except KeyboardInterrupt:
            print("👋 Thanks for using the AI Stock Chart Assistant!")
        except Exception as e:
            print(f"💥 Application error: {e}")
            messagebox.showerror("Unexpected Error", f"Something went wrong with the app: {str(e)}")

def main():
    """Main entry point for friendly version"""
    try:
        # Check Python version
        import sys
        if sys.version_info < (3, 8):
            messagebox.showerror("Python Version", "You need Python 3.8 or newer to run this app. Please update Python.")
            return
        
        print("🎯 Starting your friendly AI Stock Chart Assistant...")
        app = FriendlyStockChartAnalyzer()
        app.run()
        
    except ImportError as e:
        error_msg = f"Missing required libraries: {str(e)}\n\nPlease run: pip install -r requirements.txt"
        print(f"❌ {error_msg}")
        messagebox.showerror("Setup Required", error_msg)
    except Exception as e:
        error_msg = f"Couldn't start the application: {str(e)}"
        print(f"💥 {error_msg}")
        messagebox.showerror("Startup Error", error_msg)

if __name__ == "__main__":
    main()
