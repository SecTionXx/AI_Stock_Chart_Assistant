import customtkinter as ctk
from typing import Optional
import threading
import time
from datetime import datetime


class StatusBar(ctk.CTkFrame):
    """Bottom status bar for displaying application status and system information."""
    
    def __init__(self, parent):
        super().__init__(parent, height=40)
        
        self.current_status = "Ready"
        self.current_status_type = "info"
        
        self._setup_ui()
        self._start_clock()
        
    def _setup_ui(self):
        """Setup the status bar UI components."""
        # Configure grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_propagate(False)  # Maintain fixed height
        
        # Status icon and text
        self.status_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.status_frame.grid(row=0, column=0, sticky="w", padx=(10, 20))
        
        self.status_icon = ctk.CTkLabel(
            self.status_frame,
            text="ℹ️",
            font=ctk.CTkFont(size=14)
        )
        self.status_icon.grid(row=0, column=0, padx=(0, 5))
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Ready",
            font=ctk.CTkFont(size=12),
            anchor="w"
        )
        self.status_label.grid(row=0, column=1)
        
        # Center spacer
        spacer = ctk.CTkLabel(self, text="", fg_color="transparent")
        spacer.grid(row=0, column=1, sticky="ew")
        
        # Connection status
        self.connection_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.connection_frame.grid(row=0, column=2, sticky="e", padx=(20, 10))
        
        self.connection_icon = ctk.CTkLabel(
            self.connection_frame,
            text="🔗",
            font=ctk.CTkFont(size=12)
        )
        self.connection_icon.grid(row=0, column=0, padx=(0, 5))
        
        self.connection_label = ctk.CTkLabel(
            self.connection_frame,
            text="Offline",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.connection_label.grid(row=0, column=1, padx=(0, 15))
        
        # Clock
        self.clock_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.clock_label.grid(row=0, column=3, sticky="e", padx=(0, 10))
        
    def update_status(self, message: str, status_type: str = "info"):
        """Update the status message and icon.
        
        Args:
            message: Status message to display
            status_type: Type of status - 'info', 'success', 'warning', 'error', 'processing'
        """
        self.current_status = message
        self.current_status_type = status_type
        
        # Update icon and color based on status type
        icon_map = {
            "info": ("ℹ️", "white"),
            "success": ("✅", "green"),
            "warning": ("⚠️", "orange"),
            "error": ("❌", "red"),
            "processing": ("🔄", "orange")
        }
        
        icon, color = icon_map.get(status_type, ("ℹ️", "white"))
        
        self.status_icon.configure(text=icon)
        self.status_label.configure(text=message, text_color=color)
        
        # Auto-clear non-persistent messages after delay
        if status_type in ["success", "warning", "error"]:
            self.after(5000, lambda: self._auto_clear_status(message))
            
    def _auto_clear_status(self, original_message: str):
        """Auto-clear status if it hasn't changed."""
        if self.current_status == original_message:
            self.update_status("Ready", "info")
            
    def update_connection_status(self, connected: bool, details: str = ""):
        """Update the connection status.
        
        Args:
            connected: Whether connected to API
            details: Additional connection details
        """
        if connected:
            self.connection_icon.configure(text="🟢")
            self.connection_label.configure(
                text=f"Connected {details}".strip(),
                text_color="green"
            )
        else:
            self.connection_icon.configure(text="🔴")
            self.connection_label.configure(
                text=f"Offline {details}".strip(),
                text_color="red"
            )
            
    def set_processing(self, processing: bool, message: str = "Processing..."):
        """Set processing state with animated indicator.
        
        Args:
            processing: Whether currently processing
            message: Processing message
        """
        if processing:
            self.update_status(message, "processing")
            self._start_processing_animation()
        else:
            self._stop_processing_animation()
            
    def _start_processing_animation(self):
        """Start processing animation."""
        if not hasattr(self, '_processing_animation'):
            self._processing_animation = True
            self._animate_processing()
            
    def _stop_processing_animation(self):
        """Stop processing animation."""
        if hasattr(self, '_processing_animation'):
            self._processing_animation = False
            
    def _animate_processing(self):
        """Animate processing indicator."""
        if not hasattr(self, '_processing_animation') or not self._processing_animation:
            return
            
        # Rotate through processing icons
        icons = ["🔄", "⏳", "⌛", "🔄"]
        current_icon = self.status_icon.cget("text")
        
        try:
            current_index = icons.index(current_icon)
            next_index = (current_index + 1) % len(icons)
        except ValueError:
            next_index = 0
            
        self.status_icon.configure(text=icons[next_index])
        
        # Continue animation
        self.after(500, self._animate_processing)
        
    def _start_clock(self):
        """Start the clock update thread."""
        def update_clock():
            while True:
                try:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    self.after(0, lambda: self.clock_label.configure(text=current_time))
                    time.sleep(1)
                except:
                    break
                    
        clock_thread = threading.Thread(target=update_clock, daemon=True)
        clock_thread.start()
        
    def show_progress(self, progress: float, message: str = ""):
        """Show progress indicator.
        
        Args:
            progress: Progress value between 0.0 and 1.0
            message: Progress message
        """
        if not hasattr(self, 'progress_bar'):
            # Create progress bar if it doesn't exist
            self.progress_bar = ctk.CTkProgressBar(self, width=100, height=8)
            
        # Show progress bar
        self.progress_bar.grid(row=0, column=4, padx=(10, 5), sticky="e")
        self.progress_bar.set(progress)
        
        # Update status message
        if message:
            progress_text = f"{message} ({progress:.0%})"
            self.update_status(progress_text, "processing")
            
    def hide_progress(self):
        """Hide progress indicator."""
        if hasattr(self, 'progress_bar'):
            self.progress_bar.grid_remove()
            
    def set_api_info(self, model_name: str = "", requests_remaining: Optional[int] = None):
        """Set API information display.
        
        Args:
            model_name: Name of the AI model being used
            requests_remaining: Number of API requests remaining
        """
        info_parts = []
        
        if model_name:
            info_parts.append(f"Model: {model_name}")
            
        if requests_remaining is not None:
            info_parts.append(f"Requests: {requests_remaining}")
            
        if info_parts:
            api_info = " | ".join(info_parts)
            
            # Create or update API info label
            if not hasattr(self, 'api_info_label'):
                self.api_info_label = ctk.CTkLabel(
                    self,
                    text="",
                    font=ctk.CTkFont(size=10),
                    text_color="gray"
                )
                self.api_info_label.grid(row=0, column=5, padx=(10, 5), sticky="e")
                
            self.api_info_label.configure(text=api_info)
        else:
            # Hide API info if no information
            if hasattr(self, 'api_info_label'):
                self.api_info_label.grid_remove()
                
    def flash_message(self, message: str, status_type: str = "info", duration: int = 3000):
        """Flash a temporary message.
        
        Args:
            message: Message to flash
            status_type: Type of status
            duration: Duration in milliseconds
        """
        original_status = self.current_status
        original_type = self.current_status_type
        
        # Show flash message
        self.update_status(message, status_type)
        
        # Restore original status after duration
        self.after(duration, lambda: self.update_status(original_status, original_type))
        
    def get_current_status(self) -> tuple[str, str]:
        """Get current status message and type."""
        return self.current_status, self.current_status_type
        
    def clear_status(self):
        """Clear status to default."""
        self.update_status("Ready", "info") 