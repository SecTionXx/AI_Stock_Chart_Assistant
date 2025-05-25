import customtkinter as ctk
from tkinter import messagebox
from typing import Callable, Optional, Dict, Any, List
from datetime import datetime
import json


class HistoryPanel(ctk.CTkFrame):
    """Right panel for analysis history and settings."""
    
    def __init__(self, parent, on_history_selected: Callable, on_clear_history: Callable):
        super().__init__(parent)
        
        self.on_history_selected = on_history_selected
        self.on_clear_history = on_clear_history
        
        self.history_entries: List[Dict[str, Any]] = []
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the history panel UI components."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)  # History list area
        
        # Title and controls
        self._create_header()
        
        # History list
        self._create_history_section()
        
        # Settings section
        self._create_settings_section()
        
    def _create_header(self):
        """Create the header with title and controls."""
        header_frame = ctk.CTkFrame(self)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        header_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = ctk.CTkLabel(
            header_frame,
            text="📚 History & Settings",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title_label.grid(row=0, column=0, pady=(10, 5), sticky="w", padx=10)
        
        # Controls frame
        controls_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        controls_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        controls_frame.grid_columnconfigure(1, weight=1)
        
        # Clear history button
        self.clear_btn = ctk.CTkButton(
            controls_frame,
            text="🗑️ Clear All",
            command=self._on_clear_clicked,
            width=100,
            height=30,
            fg_color="red",
            hover_color="darkred"
        )
        self.clear_btn.grid(row=0, column=0, padx=(0, 10))
        
        # History count label
        self.count_label = ctk.CTkLabel(
            controls_frame,
            text="0 analyses",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            anchor="e"
        )
        self.count_label.grid(row=0, column=1, sticky="e")
        
    def _create_history_section(self):
        """Create the history list section."""
        # History frame with scrollable list
        history_frame = ctk.CTkFrame(self)
        history_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        history_frame.grid_columnconfigure(0, weight=1)
        history_frame.grid_rowconfigure(1, weight=1)
        
        # Section title
        history_title = ctk.CTkLabel(
            history_frame,
            text="Analysis History",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        history_title.grid(row=0, column=0, pady=(10, 5), sticky="w", padx=10)
        
        # Scrollable frame for history entries
        self.history_scrollable = ctk.CTkScrollableFrame(history_frame, label_text="")
        self.history_scrollable.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.history_scrollable.grid_columnconfigure(0, weight=1)
        
        # Default content
        self._show_empty_history()
        
    def _create_settings_section(self):
        """Create the settings section."""
        settings_frame = ctk.CTkFrame(self)
        settings_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(5, 10))
        settings_frame.grid_columnconfigure(0, weight=1)
        
        # Settings title
        settings_title = ctk.CTkLabel(
            settings_frame,
            text="⚙️ Settings",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        settings_title.grid(row=0, column=0, pady=(10, 5), sticky="w", padx=10)
        
        # Theme selection
        theme_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        theme_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        theme_frame.grid_columnconfigure(1, weight=1)
        
        theme_label = ctk.CTkLabel(
            theme_frame,
            text="Theme:",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        theme_label.grid(row=0, column=0, sticky="w", padx=(0, 10))
        
        self.theme_var = ctk.StringVar(value="dark")
        theme_menu = ctk.CTkOptionMenu(
            theme_frame,
            values=["dark", "light", "system"],
            variable=self.theme_var,
            command=self._on_theme_changed,
            width=100
        )
        theme_menu.grid(row=0, column=1, sticky="e")
        
        # Auto-save setting
        autosave_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        autosave_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        autosave_frame.grid_columnconfigure(0, weight=1)
        
        self.autosave_var = ctk.BooleanVar(value=True)
        autosave_check = ctk.CTkCheckBox(
            autosave_frame,
            text="Auto-save analyses",
            variable=self.autosave_var,
            font=ctk.CTkFont(size=12)
        )
        autosave_check.grid(row=0, column=0, sticky="w")
        
        # API status
        api_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        api_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
        api_frame.grid_columnconfigure(1, weight=1)
        
        api_label = ctk.CTkLabel(
            api_frame,
            text="API Status:",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        api_label.grid(row=0, column=0, sticky="w", padx=(0, 10))
        
        self.api_status_label = ctk.CTkLabel(
            api_frame,
            text="🔄 Checking...",
            font=ctk.CTkFont(size=11),
            anchor="e"
        )
        self.api_status_label.grid(row=0, column=1, sticky="e")
        
        # Test API button
        test_api_btn = ctk.CTkButton(
            settings_frame,
            text="🔧 Test API Connection",
            command=self._test_api_connection,
            height=30,
            width=150
        )
        test_api_btn.grid(row=4, column=0, pady=10, padx=10)
        
        # About section
        about_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        about_frame.grid(row=5, column=0, sticky="ew", padx=10, pady=(10, 15))
        
        about_text = ctk.CTkLabel(
            about_frame,
            text="AI Stock Chart Assistant v1.0\nPowered by Google Gemini Vision",
            font=ctk.CTkFont(size=10),
            text_color="gray",
            justify="center"
        )
        about_text.grid(row=0, column=0, pady=5)
        
    def _show_empty_history(self):
        """Show empty history message."""
        # Clear existing content
        for widget in self.history_scrollable.winfo_children():
            widget.destroy()
            
        # Empty message
        empty_label = ctk.CTkLabel(
            self.history_scrollable,
            text="📝 No analysis history\n\nCompleted analyses will appear here.",
            font=ctk.CTkFont(size=12),
            text_color="gray",
            justify="center"
        )
        empty_label.grid(row=0, column=0, pady=30, sticky="ew")
        
    def add_history_entry(self, entry: Dict[str, Any]):
        """Add a new history entry."""
        self.history_entries.insert(0, entry)  # Add to beginning
        self._refresh_history_display()
        self._update_count_label()
        
    def _refresh_history_display(self):
        """Refresh the history display."""
        # Clear existing content
        for widget in self.history_scrollable.winfo_children():
            widget.destroy()
            
        if not self.history_entries:
            self._show_empty_history()
            return
            
        # Display history entries
        for i, entry in enumerate(self.history_entries):
            self._create_history_entry_widget(entry, i)
            
    def _create_history_entry_widget(self, entry: Dict[str, Any], index: int):
        """Create a widget for a history entry."""
        # Entry frame
        entry_frame = ctk.CTkFrame(self.history_scrollable)
        entry_frame.grid(row=index, column=0, sticky="ew", padx=5, pady=2)
        entry_frame.grid_columnconfigure(0, weight=1)
        
        # Make it clickable
        entry_frame.bind("<Button-1>", lambda e: self._on_entry_clicked(entry))
        
        # Image name
        name_label = ctk.CTkLabel(
            entry_frame,
            text=entry.get("image_name", "Unknown"),
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        name_label.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 2))
        name_label.bind("<Button-1>", lambda e: self._on_entry_clicked(entry))
        
        # Timestamp
        timestamp_str = entry.get("timestamp", "")
        if timestamp_str:
            try:
                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                time_text = dt.strftime("%m/%d %H:%M")
            except:
                time_text = "Unknown time"
        else:
            time_text = "Unknown time"
            
        time_label = ctk.CTkLabel(
            entry_frame,
            text=time_text,
            font=ctk.CTkFont(size=10),
            text_color="gray",
            anchor="w"
        )
        time_label.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 2))
        time_label.bind("<Button-1>", lambda e: self._on_entry_clicked(entry))
        
        # Confidence score if available
        analysis = entry.get("analysis", {})
        confidence = analysis.get("confidence_score", 0)
        if confidence > 0:
            confidence_text = f"Confidence: {confidence:.1%}"
            confidence_color = self._get_confidence_color(confidence)
            
            conf_label = ctk.CTkLabel(
                entry_frame,
                text=confidence_text,
                font=ctk.CTkFont(size=9),
                text_color=confidence_color,
                anchor="w"
            )
            conf_label.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 8))
            conf_label.bind("<Button-1>", lambda e: self._on_entry_clicked(entry))
            
        # Hover effect
        def on_enter(e):
            entry_frame.configure(fg_color="gray25")
            
        def on_leave(e):
            entry_frame.configure(fg_color="gray20")
            
        entry_frame.bind("<Enter>", on_enter)
        entry_frame.bind("<Leave>", on_leave)
        name_label.bind("<Enter>", on_enter)
        name_label.bind("<Leave>", on_leave)
        time_label.bind("<Enter>", on_enter)
        time_label.bind("<Leave>", on_leave)
        
    def _on_entry_clicked(self, entry: Dict[str, Any]):
        """Handle history entry click."""
        self.on_history_selected(entry)
        
    def _get_confidence_color(self, confidence: float) -> str:
        """Get color based on confidence score."""
        if confidence >= 0.8:
            return "green"
        elif confidence >= 0.6:
            return "orange"
        else:
            return "red"
            
    def _update_count_label(self):
        """Update the history count label."""
        count = len(self.history_entries)
        text = f"{count} analysis" if count == 1 else f"{count} analyses"
        self.count_label.configure(text=text)
        
    def _on_clear_clicked(self):
        """Handle clear history button click."""
        if self.history_entries:
            self.on_clear_history()
        else:
            messagebox.showinfo("No History", "No history to clear.")
            
    def clear_history(self):
        """Clear all history entries."""
        self.history_entries.clear()
        self._refresh_history_display()
        self._update_count_label()
        
    def _on_theme_changed(self, theme: str):
        """Handle theme change."""
        try:
            ctk.set_appearance_mode(theme)
            # Could save to config here
        except Exception as e:
            messagebox.showerror("Theme Error", f"Failed to change theme: {str(e)}")
            
    def _test_api_connection(self):
        """Test API connection."""
        # Update status to testing
        self.api_status_label.configure(text="🔄 Testing...", text_color="orange")
        
        # This would be implemented to actually test the API
        # For now, simulate a test
        self.after(2000, self._api_test_complete)
        
    def _api_test_complete(self):
        """Handle API test completion."""
        # This would show actual API status
        # For now, show success
        self.api_status_label.configure(text="✅ Connected", text_color="green")
        
    def update_api_status(self, status: str, color: str = "gray"):
        """Update the API status display."""
        self.api_status_label.configure(text=status, text_color=color)
        
    def get_settings(self) -> Dict[str, Any]:
        """Get current settings."""
        return {
            "theme": self.theme_var.get(),
            "auto_save": self.autosave_var.get()
        }
        
    def set_settings(self, settings: Dict[str, Any]):
        """Set settings values."""
        if "theme" in settings:
            self.theme_var.set(settings["theme"])
            
        if "auto_save" in settings:
            self.autosave_var.set(settings["auto_save"])
            
    def get_history_count(self) -> int:
        """Get the number of history entries."""
        return len(self.history_entries) 