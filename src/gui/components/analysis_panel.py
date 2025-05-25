import customtkinter as ctk
from tkinter import messagebox
from typing import Callable, Optional, Dict, Any
import json

from ...core.analyzer import AnalysisResult


class AnalysisPanel(ctk.CTkFrame):
    """Center panel for displaying AI analysis results."""
    
    def __init__(self, parent, on_export_clicked: Callable, on_save_clicked: Callable):
        super().__init__(parent)
        
        self.on_export_clicked = on_export_clicked
        self.on_save_clicked = on_save_clicked
        
        self.current_analysis: Optional[AnalysisResult] = None
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the analysis panel UI components."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)  # Main content area
        
        # Title and controls
        self._create_header()
        
        # Main content area with scrollable frame
        self._create_content_area()
        
        # Export controls
        self._create_export_section()
        
    def _create_header(self):
        """Create the header with title and controls."""
        header_frame = ctk.CTkFrame(self)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        header_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = ctk.CTkLabel(
            header_frame,
            text="🤖 AI Analysis Results",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title_label.grid(row=0, column=0, pady=(10, 5), sticky="w", padx=10)
        
        # Controls frame
        controls_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        controls_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        controls_frame.grid_columnconfigure(1, weight=1)
        
        # Save to history button
        self.save_btn = ctk.CTkButton(
            controls_frame,
            text="💾 Save to History",
            command=self._on_save_clicked,
            width=120,
            height=30,
            state="disabled"
        )
        self.save_btn.grid(row=0, column=0, padx=(0, 10))
        
        # Confidence indicator
        self.confidence_label = ctk.CTkLabel(
            controls_frame,
            text="",
            font=ctk.CTkFont(size=11),
            anchor="e"
        )
        self.confidence_label.grid(row=0, column=1, sticky="e")
        
    def _create_content_area(self):
        """Create the scrollable content area."""
        # Scrollable frame
        self.scrollable_frame = ctk.CTkScrollableFrame(self, label_text="")
        self.scrollable_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        
        # Default content
        self._show_default_content()
        
    def _create_export_section(self):
        """Create the export controls section."""
        export_frame = ctk.CTkFrame(self)
        export_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(5, 10))
        export_frame.grid_columnconfigure(0, weight=1)
        
        # Export title
        export_title = ctk.CTkLabel(
            export_frame,
            text="Export Options",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        export_title.grid(row=0, column=0, pady=(10, 5), sticky="w", padx=10)
        
        # Export buttons frame
        buttons_frame = ctk.CTkFrame(export_frame, fg_color="transparent")
        buttons_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        buttons_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        # Export buttons
        self.export_pdf_btn = ctk.CTkButton(
            buttons_frame,
            text="📄 PDF",
            command=lambda: self._on_export_clicked("pdf"),
            width=80,
            height=30,
            state="disabled"
        )
        self.export_pdf_btn.grid(row=0, column=0, padx=2)
        
        self.export_txt_btn = ctk.CTkButton(
            buttons_frame,
            text="📝 Text",
            command=lambda: self._on_export_clicked("txt"),
            width=80,
            height=30,
            state="disabled"
        )
        self.export_txt_btn.grid(row=0, column=1, padx=2)
        
        self.export_json_btn = ctk.CTkButton(
            buttons_frame,
            text="🔧 JSON",
            command=lambda: self._on_export_clicked("json"),
            width=80,
            height=30,
            state="disabled"
        )
        self.export_json_btn.grid(row=0, column=2, padx=2)
        
    def _show_default_content(self):
        """Show default content when no analysis is available."""
        # Clear existing content
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        # Default message
        default_label = ctk.CTkLabel(
            self.scrollable_frame,
            text="🔍 No analysis available\n\nSelect an image and click 'Analyze Chart' to begin AI analysis.",
            font=ctk.CTkFont(size=14),
            text_color="gray",
            justify="center"
        )
        default_label.grid(row=0, column=0, pady=50, sticky="ew")
        
    def display_analysis(self, analysis: AnalysisResult):
        """Display the analysis results."""
        self.current_analysis = analysis
        
        # Clear existing content
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        if not analysis.success:
            self._show_error_content(analysis)
            return
            
        # Update confidence indicator
        confidence_text = f"Confidence: {analysis.confidence_score:.1%}"
        confidence_color = self._get_confidence_color(analysis.confidence_score)
        self.confidence_label.configure(text=confidence_text, text_color=confidence_color)
        
        # Create analysis sections
        row = 0
        
        # Summary section
        if analysis.analysis_text:
            row = self._create_summary_section(analysis.analysis_text, row)
            
        # Technical indicators section
        if analysis.technical_indicators:
            row = self._create_indicators_section(analysis.technical_indicators, row)
            
        # Recommendations section
        if analysis.recommendations:
            row = self._create_recommendations_section(analysis.recommendations, row)
            
        # Performance info
        if analysis.processing_time:
            row = self._create_performance_section(analysis.processing_time, row)
            
        # Enable controls
        self._enable_controls()
        
    def _show_error_content(self, analysis: AnalysisResult):
        """Show error content when analysis failed."""
        error_label = ctk.CTkLabel(
            self.scrollable_frame,
            text="❌ Analysis Failed\n\nThere was an error analyzing the image. Please try again.",
            font=ctk.CTkFont(size=14),
            text_color="red",
            justify="center"
        )
        error_label.grid(row=0, column=0, pady=50, sticky="ew")
        
        # Disable controls
        self._disable_controls()
        
    def _create_summary_section(self, analysis_text: str, start_row: int) -> int:
        """Create the analysis summary section."""
        # Section title
        summary_title = ctk.CTkLabel(
            self.scrollable_frame,
            text="📊 Analysis Summary",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w"
        )
        summary_title.grid(row=start_row, column=0, sticky="ew", padx=10, pady=(10, 5))
        
        # Summary text
        summary_frame = ctk.CTkFrame(self.scrollable_frame)
        summary_frame.grid(row=start_row+1, column=0, sticky="ew", padx=10, pady=(0, 15))
        summary_frame.grid_columnconfigure(0, weight=1)
        
        summary_text = ctk.CTkTextbox(
            summary_frame,
            height=120,
            font=ctk.CTkFont(size=12),
            wrap="word"
        )
        summary_text.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        summary_text.insert("1.0", analysis_text)
        summary_text.configure(state="disabled")
        
        return start_row + 2
        
    def _create_indicators_section(self, indicators: Dict[str, Any], start_row: int) -> int:
        """Create the technical indicators section."""
        # Section title
        indicators_title = ctk.CTkLabel(
            self.scrollable_frame,
            text="📈 Technical Indicators",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w"
        )
        indicators_title.grid(row=start_row, column=0, sticky="ew", padx=10, pady=(10, 5))
        
        # Indicators frame
        indicators_frame = ctk.CTkFrame(self.scrollable_frame)
        indicators_frame.grid(row=start_row+1, column=0, sticky="ew", padx=10, pady=(0, 15))
        indicators_frame.grid_columnconfigure(1, weight=1)
        
        # Display indicators
        row = 0
        for key, value in indicators.items():
            # Format key
            display_key = key.replace('_', ' ').title()
            
            # Key label
            key_label = ctk.CTkLabel(
                indicators_frame,
                text=f"{display_key}:",
                font=ctk.CTkFont(size=12, weight="bold"),
                anchor="w"
            )
            key_label.grid(row=row, column=0, sticky="w", padx=(10, 5), pady=2)
            
            # Value label
            value_text = str(value)
            if isinstance(value, (int, float)):
                if key.lower() in ['price', 'support', 'resistance']:
                    value_text = f"${value:.2f}"
                elif key.lower() in ['volume']:
                    value_text = f"{value:,}"
                else:
                    value_text = f"{value:.2f}"
                    
            value_label = ctk.CTkLabel(
                indicators_frame,
                text=value_text,
                font=ctk.CTkFont(size=12),
                anchor="w"
            )
            value_label.grid(row=row, column=1, sticky="w", padx=(5, 10), pady=2)
            
            row += 1
            
        # Add padding
        if row == 0:
            no_data_label = ctk.CTkLabel(
                indicators_frame,
                text="No technical indicators available",
                font=ctk.CTkFont(size=11),
                text_color="gray"
            )
            no_data_label.grid(row=0, column=0, columnspan=2, pady=10)
        else:
            ctk.CTkLabel(indicators_frame, text="").grid(row=row, column=0, pady=(0, 10))
            
        return start_row + 2
        
    def _create_recommendations_section(self, recommendations: list, start_row: int) -> int:
        """Create the recommendations section."""
        # Section title
        rec_title = ctk.CTkLabel(
            self.scrollable_frame,
            text="💡 Recommendations",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w"
        )
        rec_title.grid(row=start_row, column=0, sticky="ew", padx=10, pady=(10, 5))
        
        # Recommendations frame
        rec_frame = ctk.CTkFrame(self.scrollable_frame)
        rec_frame.grid(row=start_row+1, column=0, sticky="ew", padx=10, pady=(0, 15))
        rec_frame.grid_columnconfigure(0, weight=1)
        
        # Display recommendations
        if recommendations:
            for i, rec in enumerate(recommendations):
                rec_label = ctk.CTkLabel(
                    rec_frame,
                    text=f"• {rec}",
                    font=ctk.CTkFont(size=12),
                    anchor="w",
                    justify="left",
                    wraplength=400
                )
                rec_label.grid(row=i, column=0, sticky="ew", padx=10, pady=2)
        else:
            no_rec_label = ctk.CTkLabel(
                rec_frame,
                text="No specific recommendations available",
                font=ctk.CTkFont(size=11),
                text_color="gray"
            )
            no_rec_label.grid(row=0, column=0, pady=10)
            
        # Add padding
        ctk.CTkLabel(rec_frame, text="").grid(row=len(recommendations), column=0, pady=(0, 10))
        
        return start_row + 2
        
    def _create_performance_section(self, processing_time: float, start_row: int) -> int:
        """Create the performance information section."""
        # Performance info
        perf_label = ctk.CTkLabel(
            self.scrollable_frame,
            text=f"⏱️ Analysis completed in {processing_time:.2f} seconds",
            font=ctk.CTkFont(size=10),
            text_color="gray",
            anchor="w"
        )
        perf_label.grid(row=start_row, column=0, sticky="ew", padx=10, pady=(5, 10))
        
        return start_row + 1
        
    def _get_confidence_color(self, confidence: float) -> str:
        """Get color based on confidence score."""
        if confidence >= 0.8:
            return "green"
        elif confidence >= 0.6:
            return "orange"
        else:
            return "red"
            
    def _enable_controls(self):
        """Enable analysis-related controls."""
        self.save_btn.configure(state="normal")
        self.export_pdf_btn.configure(state="normal")
        self.export_txt_btn.configure(state="normal")
        self.export_json_btn.configure(state="normal")
        
    def _disable_controls(self):
        """Disable analysis-related controls."""
        self.save_btn.configure(state="disabled")
        self.export_pdf_btn.configure(state="disabled")
        self.export_txt_btn.configure(state="disabled")
        self.export_json_btn.configure(state="disabled")
        self.confidence_label.configure(text="")
        
    def _on_save_clicked(self):
        """Handle save to history button click."""
        if self.current_analysis:
            self.on_save_clicked()
        else:
            messagebox.showwarning("No Analysis", "No analysis to save.")
            
    def _on_export_clicked(self, export_format: str):
        """Handle export button click."""
        if self.current_analysis:
            self.on_export_clicked(export_format)
        else:
            messagebox.showwarning("No Analysis", "No analysis to export.")
            
    def clear_analysis(self):
        """Clear the current analysis display."""
        self.current_analysis = None
        self._show_default_content()
        self._disable_controls()
        
    def get_current_analysis(self) -> Optional[AnalysisResult]:
        """Get the current analysis result."""
        return self.current_analysis 