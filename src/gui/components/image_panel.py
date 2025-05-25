import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from pathlib import Path
from typing import Callable, Optional
import threading

from ...utils.image_handler import ImageInfo


class ImagePanel(ctk.CTkFrame):
    """Left panel for image upload, preview, and analysis controls."""
    
    def __init__(self, parent, on_image_selected: Callable, on_analyze_clicked: Callable):
        super().__init__(parent)
        
        self.on_image_selected = on_image_selected
        self.on_analyze_clicked = on_analyze_clicked
        
        self.current_image: Optional[Image.Image] = None
        self.current_image_path: Optional[Path] = None
        self.image_info: Optional[ImageInfo] = None
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the image panel UI components."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)  # Image preview area
        
        # Title
        title_label = ctk.CTkLabel(
            self, 
            text="📊 Image Upload", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title_label.grid(row=0, column=0, pady=(10, 20), sticky="ew")
        
        # Upload section
        self._create_upload_section()
        
        # Image preview section
        self._create_preview_section()
        
        # Image info section
        self._create_info_section()
        
        # Analysis controls
        self._create_analysis_section()
        
    def _create_upload_section(self):
        """Create the upload controls section."""
        upload_frame = ctk.CTkFrame(self)
        upload_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        upload_frame.grid_columnconfigure(0, weight=1)
        
        # Upload button
        self.upload_btn = ctk.CTkButton(
            upload_frame,
            text="📁 Select Image",
            command=self._select_image,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.upload_btn.grid(row=0, column=0, pady=10, padx=10, sticky="ew")
        
        # Drag and drop label
        drop_label = ctk.CTkLabel(
            upload_frame,
            text="or drag & drop image here",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        drop_label.grid(row=1, column=0, pady=(0, 10))
        
        # Supported formats
        formats_label = ctk.CTkLabel(
            upload_frame,
            text="Supported: PNG, JPG, JPEG, GIF, BMP\nMax size: 20MB",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        formats_label.grid(row=2, column=0, pady=(0, 10))
        
    def _create_preview_section(self):
        """Create the image preview section."""
        preview_frame = ctk.CTkFrame(self)
        preview_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(0, weight=1)
        
        # Preview label
        preview_title = ctk.CTkLabel(
            preview_frame,
            text="Image Preview",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        preview_title.grid(row=0, column=0, pady=(10, 5), sticky="ew")
        
        # Image display area
        self.image_label = ctk.CTkLabel(
            preview_frame,
            text="No image selected",
            width=300,
            height=200,
            fg_color="gray20",
            corner_radius=8
        )
        self.image_label.grid(row=1, column=0, pady=10, padx=10, sticky="nsew")
        
        # Image name label
        self.image_name_label = ctk.CTkLabel(
            preview_frame,
            text="",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.image_name_label.grid(row=2, column=0, pady=(0, 10))
        
    def _create_info_section(self):
        """Create the image information section."""
        info_frame = ctk.CTkFrame(self)
        info_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))
        info_frame.grid_columnconfigure(1, weight=1)
        
        # Info title
        info_title = ctk.CTkLabel(
            info_frame,
            text="Image Information",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        info_title.grid(row=0, column=0, columnspan=2, pady=(10, 5), sticky="ew")
        
        # Info labels
        self.info_labels = {}
        info_items = [
            ("Size:", "size_label"),
            ("Dimensions:", "dimensions_label"),
            ("Format:", "format_label"),
            ("Status:", "status_label")
        ]
        
        for i, (label_text, label_key) in enumerate(info_items, 1):
            # Label
            label = ctk.CTkLabel(
                info_frame,
                text=label_text,
                font=ctk.CTkFont(size=11, weight="bold"),
                anchor="w"
            )
            label.grid(row=i, column=0, sticky="w", padx=(10, 5), pady=2)
            
            # Value
            value_label = ctk.CTkLabel(
                info_frame,
                text="-",
                font=ctk.CTkFont(size=11),
                anchor="w"
            )
            value_label.grid(row=i, column=1, sticky="w", padx=(5, 10), pady=2)
            
            self.info_labels[label_key] = value_label
            
        # Add bottom padding
        ctk.CTkLabel(info_frame, text="").grid(row=len(info_items)+1, column=0, pady=(0, 10))
        
    def _create_analysis_section(self):
        """Create the analysis controls section."""
        analysis_frame = ctk.CTkFrame(self)
        analysis_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 10))
        analysis_frame.grid_columnconfigure(0, weight=1)
        
        # Analysis title
        analysis_title = ctk.CTkLabel(
            analysis_frame,
            text="AI Analysis",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        analysis_title.grid(row=0, column=0, pady=(10, 5), sticky="ew")
        
        # Analyze button
        self.analyze_btn = ctk.CTkButton(
            analysis_frame,
            text="🤖 Analyze Chart",
            command=self._on_analyze_clicked,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled"
        )
        self.analyze_btn.grid(row=1, column=0, pady=10, padx=10, sticky="ew")
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(analysis_frame)
        self.progress_bar.grid(row=2, column=0, pady=(0, 10), padx=10, sticky="ew")
        self.progress_bar.set(0)
        self.progress_bar.grid_remove()  # Hide initially
        
        # Status label
        self.analysis_status_label = ctk.CTkLabel(
            analysis_frame,
            text="Select an image to begin analysis",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.analysis_status_label.grid(row=3, column=0, pady=(0, 10))
        
    def _select_image(self):
        """Open file dialog to select an image."""
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
            self.load_image(Path(filename))
            
    def load_image(self, image_path: Path):
        """Load and display an image."""
        try:
            self.current_image_path = image_path
            
            # Load image
            image = Image.open(image_path)
            self.current_image = image
            
            # Display image
            self._display_image(image)
            
            # Update image name
            self.image_name_label.configure(text=image_path.name)
            
            # Enable analyze button
            self.analyze_btn.configure(state="normal")
            self.analysis_status_label.configure(text="Ready for analysis")
            
            # Notify parent
            self.on_image_selected(image_path)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            
    def _display_image(self, image: Image.Image):
        """Display image in the preview area."""
        try:
            # Calculate display size while maintaining aspect ratio
            display_width = 280
            display_height = 180
            
            # Calculate scaling
            img_width, img_height = image.size
            scale_w = display_width / img_width
            scale_h = display_height / img_height
            scale = min(scale_w, scale_h)
            
            # Resize image
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            display_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(display_image)
            
            # Update label
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            self.image_label.configure(text=f"Error displaying image: {str(e)}")
            
    def update_image_info(self, image_info: ImageInfo):
        """Update the image information display."""
        self.image_info = image_info
        
        # Update info labels
        if image_info.size_bytes:
            size_mb = image_info.size_bytes / (1024 * 1024)
            self.info_labels["size_label"].configure(text=f"{size_mb:.1f} MB")
        else:
            self.info_labels["size_label"].configure(text="-")
            
        if image_info.dimensions:
            width, height = image_info.dimensions
            self.info_labels["dimensions_label"].configure(text=f"{width} × {height}")
        else:
            self.info_labels["dimensions_label"].configure(text="-")
            
        self.info_labels["format_label"].configure(text=image_info.format or "-")
        
        # Status with color coding
        if image_info.is_valid:
            self.info_labels["status_label"].configure(
                text="✅ Valid", 
                text_color="green"
            )
        else:
            error_text = "❌ " + (image_info.error_message if image_info.error_message else "Invalid")
            self.info_labels["status_label"].configure(
                text=error_text,
                text_color="red"
            )
            
    def set_analyzing(self, analyzing: bool):
        """Set the analyzing state."""
        if analyzing:
            self.analyze_btn.configure(
                text="🔄 Analyzing...",
                state="disabled"
            )
            self.progress_bar.grid()
            self.progress_bar.start()
            self.analysis_status_label.configure(
                text="AI is analyzing your chart...",
                text_color="orange"
            )
        else:
            self.analyze_btn.configure(
                text="🤖 Analyze Chart",
                state="normal" if self.current_image_path else "disabled"
            )
            self.progress_bar.stop()
            self.progress_bar.grid_remove()
            self.analysis_status_label.configure(
                text="Ready for analysis",
                text_color="gray"
            )
            
    def _on_analyze_clicked(self):
        """Handle analyze button click."""
        if self.current_image_path:
            self.on_analyze_clicked()
        else:
            messagebox.showwarning("No Image", "Please select an image first.")
            
    def clear_image(self):
        """Clear the current image."""
        self.current_image = None
        self.current_image_path = None
        self.image_info = None
        
        # Reset UI
        self.image_label.configure(image="", text="No image selected")
        self.image_name_label.configure(text="")
        self.analyze_btn.configure(state="disabled")
        self.analysis_status_label.configure(text="Select an image to begin analysis")
        
        # Reset info labels
        for label in self.info_labels.values():
            label.configure(text="-", text_color="white")
            
    def get_current_image_path(self) -> Optional[Path]:
        """Get the current image path."""
        return self.current_image_path 