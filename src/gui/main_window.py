import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

from ..core.config import get_config
from ..core.error_handler import get_error_handler, ErrorCategory
from ..core.analyzer import get_analyzer
from ..utils.image_handler import get_image_processor
from .components.image_panel import ImagePanel
from .components.analysis_panel import AnalysisPanel
from .components.history_panel import HistoryPanel
from .components.status_bar import StatusBar


class MainWindow:
    """Main application window with three-column layout and professional UI."""
    
    def __init__(self):
        self.config = get_config()
        self.error_handler = get_error_handler()
        self.analyzer = get_analyzer()
        self.image_processor = get_image_processor()
        
        # Initialize UI
        self._setup_window()
        self._create_layout()
        self._setup_bindings()
        
        # State management
        self.current_image_path: Optional[Path] = None
        self.current_analysis: Optional[Dict[str, Any]] = None
        self.analysis_history: list = []
        
        # Load session if available
        self._load_session()
        
    def _setup_window(self):
        """Initialize the main window with professional styling."""
        # Set appearance mode and color theme
        ctk.set_appearance_mode(self.config.ui.theme)
        ctk.set_default_color_theme(self.config.ui.color_theme)
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("AI Stock Chart Assistant")
        self.root.geometry(f"{self.config.ui.window_width}x{self.config.ui.window_height}")
        self.root.minsize(1000, 700)
        
        # Configure grid weights for responsive design
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Set window icon if available
        try:
            icon_path = Path("assets/icon.ico")
            if icon_path.exists():
                self.root.iconbitmap(str(icon_path))
        except Exception:
            pass  # Icon not critical
            
    def _create_layout(self):
        """Create the three-column layout with panels."""
        # Left Panel - Image Upload and Preview
        self.image_panel = ImagePanel(
            self.root,
            on_image_selected=self._on_image_selected,
            on_analyze_clicked=self._on_analyze_clicked
        )
        self.image_panel.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        
        # Center Panel - Analysis Results
        self.analysis_panel = AnalysisPanel(
            self.root,
            on_export_clicked=self._on_export_clicked,
            on_save_clicked=self._on_save_to_history
        )
        self.analysis_panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=10)
        
        # Right Panel - History and Settings
        self.history_panel = HistoryPanel(
            self.root,
            on_history_selected=self._on_history_selected,
            on_clear_history=self._on_clear_history
        )
        self.history_panel.grid(row=0, column=2, sticky="nsew", padx=(5, 10), pady=10)
        
        # Status Bar
        self.status_bar = StatusBar(self.root)
        self.status_bar.grid(row=1, column=0, columnspan=3, sticky="ew", padx=10, pady=(0, 10))
        
        # Update status
        self.status_bar.update_status("Ready", "info")
        
    def _setup_bindings(self):
        """Setup keyboard shortcuts and window events."""
        # Keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self._open_file_dialog())
        self.root.bind('<Control-s>', lambda e: self._save_session())
        self.root.bind('<Control-q>', lambda e: self._on_closing())
        self.root.bind('<F5>', lambda e: self._refresh_analysis())
        
        # Window events
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Drag and drop (if supported)
        try:
            self.root.drop_target_register('DND_Files')
            self.root.dnd_bind('<<Drop>>', self._on_file_drop)
        except Exception:
            pass  # Drag and drop not critical
            
    def _on_image_selected(self, image_path: Path):
        """Handle image selection from the image panel."""
        try:
            self.current_image_path = image_path
            self.status_bar.update_status(f"Image loaded: {image_path.name}", "success")
            
            # Validate image
            image_info = self.image_processor.validate_image(image_path)
            if not image_info.is_valid:
                self.error_handler.handle_error(
                    ValueError(f"Invalid image: {image_info.error_message}"),
                    f"Image validation failed for: {image_path}"
                )
                return
                
            # Update image panel with validation info
            self.image_panel.update_image_info(image_info)
            
            # Clear previous analysis
            self.analysis_panel.clear_analysis()
            self.current_analysis = None
            
        except Exception as e:
            self.error_handler.handle_error(
                e, f"Error loading image: {image_path}"
            )
            
    def _on_analyze_clicked(self):
        """Handle analyze button click."""
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please select an image first.")
            return
            
        # Start analysis in background thread
        self.status_bar.update_status("Analyzing image...", "processing")
        self.image_panel.set_analyzing(True)
        
        thread = threading.Thread(target=self._perform_analysis, daemon=True)
        thread.start()
        
    def _perform_analysis(self):
        """Perform AI analysis in background thread."""
        try:
            # Analyze image
            result = self.analyzer.analyze_chart(
                str(self.current_image_path),
                analysis_type="comprehensive"
            )
            
            # Update UI in main thread
            self.root.after(0, self._on_analysis_complete, result)
            
        except Exception as e:
            self.root.after(0, self._on_analysis_error, e)
            
    def _on_analysis_complete(self, result):
        """Handle successful analysis completion."""
        self.current_analysis = result.to_dict()
        self.analysis_panel.display_analysis(result)
        self.image_panel.set_analyzing(False)
        self.status_bar.update_status("Analysis complete", "success")
        
        # Auto-save session
        self._save_session()
        
    def _on_analysis_error(self, error):
        """Handle analysis error."""
        self.error_handler.handle_error(
            error, f"API analysis error for image: {self.current_image_path}"
        )
        self.image_panel.set_analyzing(False)
        self.status_bar.update_status("Analysis failed", "error")
        
    def _on_export_clicked(self, export_format: str):
        """Handle export button click."""
        if not self.current_analysis:
            messagebox.showwarning("No Analysis", "No analysis to export.")
            return
            
        try:
            # Get export path
            file_types = {
                "pdf": [("PDF files", "*.pdf")],
                "txt": [("Text files", "*.txt")],
                "json": [("JSON files", "*.json")]
            }
            
            filename = filedialog.asksaveasfilename(
                defaultextension=f".{export_format}",
                filetypes=file_types.get(export_format, [("All files", "*.*")])
            )
            
            if filename:
                # Create AnalysisResult object from current analysis
                from ..core.analyzer import AnalysisResult
                
                result = AnalysisResult(
                    success=True,
                    analysis_text=self.current_analysis.get("analysis_text", ""),
                    technical_indicators=self.current_analysis.get("technical_indicators", {}),
                    recommendations=self.current_analysis.get("recommendations", []),
                    confidence_score=self.current_analysis.get("confidence_score", 0.0),
                    processing_time=self.current_analysis.get("processing_time", 0.0),
                    risk_assessment=self.current_analysis.get("risk_assessment", "Unknown"),
                    timestamp=datetime.now(),
                    image_path=str(self.current_image_path) if self.current_image_path else None
                )
                
                # Export analysis
                export_path = self.analyzer.export_analysis(result, export_format)
                
                if export_path:
                    self.status_bar.update_status(f"Exported to {Path(export_path).name}", "success")
                    messagebox.showinfo("Export Complete", f"Analysis exported to {export_path}")
                else:
                    messagebox.showerror("Export Failed", "Failed to export analysis.")
                    
        except Exception as e:
            self.error_handler.handle_error(
                e, f"Export failed for format: {export_format}"
            )
            
    def _on_save_to_history(self):
        """Save current analysis to history."""
        if not self.current_analysis:
            messagebox.showwarning("No Analysis", "No analysis to save.")
            return
            
        try:
            # Add timestamp and image info
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "image_path": str(self.current_image_path),
                "image_name": self.current_image_path.name,
                "analysis": self.current_analysis
            }
            
            self.analysis_history.append(history_entry)
            self.history_panel.add_history_entry(history_entry)
            
            # Save to file
            self._save_history()
            
            self.status_bar.update_status("Analysis saved to history", "success")
            
        except Exception as e:
            self.error_handler.handle_error(
                e, "Failed to save analysis to history"
            )
            
    def _on_history_selected(self, entry):
        """Handle history entry selection."""
        try:
            self.current_analysis = entry["analysis"]
            
            # Create analysis result object
            from ..core.analyzer import AnalysisResult
            result = AnalysisResult(
                success=True,
                analysis_text=entry["analysis"].get("analysis_text", ""),
                technical_indicators=entry["analysis"].get("technical_indicators", {}),
                recommendations=entry["analysis"].get("recommendations", []),
                confidence_score=entry["analysis"].get("confidence_score", 0.0),
                processing_time=entry["analysis"].get("processing_time", 0.0)
            )
            
            self.analysis_panel.display_analysis(result)
            
            # Try to load the image if it still exists
            image_path = Path(entry["image_path"])
            if image_path.exists():
                self.current_image_path = image_path
                self.image_panel.load_image(image_path)
                
            self.status_bar.update_status(f"Loaded history: {entry['image_name']}", "info")
            
        except Exception as e:
            self.error_handler.handle_error(
                e, "Failed to load history entry"
            )
            
    def _on_clear_history(self):
        """Clear analysis history."""
        if messagebox.askyesno("Clear History", "Are you sure you want to clear all history?"):
            self.analysis_history.clear()
            self.history_panel.clear_history()
            self._save_history()
            self.status_bar.update_status("History cleared", "info")
            
    def _open_file_dialog(self):
        """Open file dialog to select image."""
        file_types = [
            ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Stock Chart Image",
            filetypes=file_types
        )
        
        if filename:
            self.image_panel.load_image(Path(filename))
            
    def _on_file_drop(self, event):
        """Handle file drop event."""
        try:
            files = event.data.split()
            if files:
                file_path = Path(files[0])
                if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                    self.image_panel.load_image(file_path)
        except Exception as e:
            self.error_handler.handle_error(
                e, "File drop operation failed"
            )
            
    def _refresh_analysis(self):
        """Refresh current analysis."""
        if self.current_image_path:
            self._on_analyze_clicked()
            
    def _save_session(self):
        """Save current session state."""
        try:
            session_data = {
                "current_image_path": str(self.current_image_path) if self.current_image_path else None,
                "current_analysis": self.current_analysis,
                "window_geometry": self.root.geometry(),
                "timestamp": datetime.now().isoformat()
            }
            
            session_file = self.config.storage.sessions_dir / "current_session.json"
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
                
        except Exception as e:
            self.error_handler.handle_error(
                e, "Failed to save session"
            )
            
    def _load_session(self):
        """Load previous session if available."""
        try:
            session_file = self.config.storage.sessions_dir / "current_session.json"
            if session_file.exists():
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                    
                # Restore window geometry
                if "window_geometry" in session_data:
                    self.root.geometry(session_data["window_geometry"])
                    
                # Restore image and analysis
                if session_data.get("current_image_path"):
                    image_path = Path(session_data["current_image_path"])
                    if image_path.exists():
                        self.current_image_path = image_path
                        self.image_panel.load_image(image_path)
                        
                if session_data.get("current_analysis"):
                    self.current_analysis = session_data["current_analysis"]
                    # Restore analysis display would need AnalysisResult reconstruction
                    
        except Exception as e:
            # Session loading is not critical
            pass
            
    def _save_history(self):
        """Save analysis history to file."""
        try:
            history_file = self.config.storage.history_dir / "analysis_history.json"
            with open(history_file, 'w') as f:
                json.dump(self.analysis_history, f, indent=2)
        except Exception as e:
            self.error_handler.handle_error(
                e, "Failed to save analysis history"
            )
            
    def _load_history(self):
        """Load analysis history from file."""
        try:
            history_file = self.config.storage.history_dir / "analysis_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.analysis_history = json.load(f)
                    
                # Populate history panel
                for entry in self.analysis_history:
                    self.history_panel.add_history_entry(entry)
                    
        except Exception as e:
            self.error_handler.handle_error(
                e, "Failed to load analysis history"
            )
            
    def _on_closing(self):
        """Handle window closing event."""
        try:
            # Save session
            self._save_session()
            
            # Save history
            self._save_history()
            
            # Cleanup
            self.error_handler.cleanup()
            
        except Exception:
            pass  # Don't prevent closing
        finally:
            self.root.destroy()
            
    def run(self):
        """Start the application."""
        # Load history
        self._load_history()
        
        # Start the main loop
        self.root.mainloop() 