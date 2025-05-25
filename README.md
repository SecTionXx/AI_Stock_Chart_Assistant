# 🤖 AI Stock Chart Assistant

A production-ready AI-powered stock chart analysis tool with enhanced error handling, session recovery, and professional user interface.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)

## ✨ Features

### 🎯 Core Functionality
- **AI-Powered Analysis**: Google Gemini Vision API integration for intelligent stock chart analysis
- **Professional GUI**: Modern CustomTkinter interface with three-column layout
- **Real-time Processing**: Background analysis with progress indicators and status updates
- **Multi-format Support**: PNG, JPG, JPEG, GIF, BMP image formats (up to 20MB)

### 🛡️ Reliability & Error Handling
- **Advanced Error Handling**: Comprehensive error categorization and user-friendly messages
- **Automatic Retry**: Exponential backoff for API calls and network operations
- **Session Recovery**: Auto-save and restore application state
- **Health Monitoring**: Real-time API connection and system health checks
- **Graceful Degradation**: Offline mode with cached results

### 📊 Analysis & Export
- **Technical Indicators**: Support levels, resistance, trends, volume analysis
- **Smart Recommendations**: AI-generated trading insights and suggestions
- **Confidence Scoring**: Analysis reliability indicators
- **Multiple Export Formats**: PDF, Text, and JSON export options
- **Analysis History**: Persistent storage with searchable history

### 🎨 User Experience
- **Drag & Drop**: Easy image upload with validation
- **Keyboard Shortcuts**: Power user productivity features
- **Theme Support**: Dark, light, and system themes
- **Status Indicators**: Real-time feedback and progress tracking
- **Responsive Design**: Adaptive layout for different screen sizes

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-stock-chart-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_new.txt
   ```

3. **Setup environment variables**
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key_here
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

## 📖 Usage Guide

### Getting Started
1. **Launch the application** using `python main.py`
2. **Upload an image** by clicking "Select Image" or dragging and dropping
3. **Analyze the chart** by clicking "Analyze Chart"
4. **Review results** in the center panel
5. **Save or export** your analysis as needed

### Keyboard Shortcuts
- `Ctrl+O`: Open image file dialog
- `Ctrl+S`: Save current session
- `F5`: Refresh/re-run analysis
- `Ctrl+Q`: Quit application

### Image Requirements
- **Formats**: PNG, JPG, JPEG, GIF, BMP
- **Size**: Maximum 20MB
- **Content**: Stock charts, candlestick patterns, technical analysis charts
- **Quality**: Higher resolution images provide better analysis results

## 🏗️ Project Structure

```
ai-stock-chart-assistant/
├── src/
│   ├── core/                    # Core application logic
│   │   ├── config.py           # Configuration management
│   │   ├── error_handler.py    # Error handling system
│   │   └── analyzer.py         # AI analysis engine
│   ├── gui/                    # User interface components
│   │   ├── main_window.py      # Main application window
│   │   └── components/         # UI components
│   │       ├── image_panel.py  # Image upload/preview
│   │       ├── analysis_panel.py # Results display
│   │       ├── history_panel.py # History/settings
│   │       └── status_bar.py   # Status information
│   └── utils/                  # Utility modules
│       └── image_handler.py    # Image processing
├── data/                       # Application data
│   ├── sessions/              # Session backups
│   ├── history/               # Analysis history
│   ├── exports/               # Exported files
│   └── logs/                  # Application logs
├── tests/                     # Test suite
├── requirements_new.txt       # Dependencies
├── main.py                   # Application entry point
└── README.md                 # This file
```

## 🔧 Configuration

The application uses a hierarchical configuration system:

### Environment Variables
- `GEMINI_API_KEY`: Google Gemini API key (required)
- `LOG_LEVEL`: Logging level (default: INFO)
- `DATA_DIR`: Data directory path (default: ./data)

### Configuration Files
- `.env`: Environment variables
- `data/sessions/`: Session state files
- `data/logs/`: Application logs

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Run with coverage:
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

## 📊 API Integration

### Google Gemini Vision API
The application uses Google's Gemini Vision API for chart analysis:

- **Model**: gemini-pro-vision
- **Rate Limiting**: Built-in request throttling
- **Error Handling**: Automatic retry with exponential backoff
- **Prompt Engineering**: Optimized prompts for financial chart analysis

### Analysis Capabilities
- **Chart Pattern Recognition**: Candlestick patterns, trends, formations
- **Technical Indicators**: Moving averages, support/resistance levels
- **Volume Analysis**: Trading volume patterns and significance
- **Market Sentiment**: Bullish/bearish indicators and signals

## 🛠️ Development

### Setting up Development Environment
1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements_new.txt`
5. Install development dependencies: `pip install pytest pytest-cov black flake8`

### Code Style
- **Formatter**: Black
- **Linter**: Flake8
- **Type Hints**: Encouraged throughout
- **Docstrings**: Google style

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📝 Dependencies

### Core Dependencies
- **customtkinter**: Modern GUI framework
- **google-generativeai**: Google Gemini API client
- **Pillow**: Image processing
- **opencv-python**: Computer vision features
- **tenacity**: Retry mechanisms
- **structlog**: Structured logging
- **python-dotenv**: Environment variable management

### Optional Dependencies
- **reportlab**: PDF export functionality
- **matplotlib**: Chart visualization
- **pandas**: Data analysis features

## 🔒 Security & Privacy

- **API Keys**: Stored securely in environment variables
- **Local Processing**: Images processed locally before API submission
- **No Data Collection**: No user data is collected or transmitted
- **Session Encryption**: Session files can be encrypted (optional)

## 🐛 Troubleshooting

### Common Issues

**"Missing API Key" Error**
- Ensure `GEMINI_API_KEY` is set in your `.env` file
- Verify the API key is valid and has proper permissions

**"Dependencies Missing" Error**
- Run `pip install -r requirements_new.txt`
- Ensure you're using Python 3.8 or higher

**"Image Validation Failed" Error**
- Check image format (PNG, JPG, JPEG, GIF, BMP)
- Ensure image size is under 20MB
- Verify image is not corrupted

**GUI Not Displaying Correctly**
- Update CustomTkinter: `pip install --upgrade customtkinter`
- Check system theme compatibility
- Try different appearance modes in settings

### Getting Help
1. Check the logs in `data/logs/` for detailed error information
2. Use the "Test API Connection" button in settings
3. Review the troubleshooting section above
4. Submit an issue with logs and system information

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini**: For providing the AI vision capabilities
- **CustomTkinter**: For the modern GUI framework
- **OpenCV**: For image processing capabilities
- **Python Community**: For the excellent ecosystem of libraries

## 🔮 Roadmap

### Phase 2 Features (Planned)
- **Batch Processing**: Analyze multiple charts simultaneously
- **Custom Templates**: User-defined analysis templates
- **Drawing Tools**: Annotate charts with trend lines and markers
- **Advanced Filters**: Filter analysis history by various criteria

### Phase 3 Features (Future)
- **Real-time Data**: Integration with live market data feeds
- **Portfolio Tracking**: Track multiple stocks and portfolios
- **Alert System**: Notifications for significant chart patterns
- **Mobile App**: Companion mobile application

---

**Made with ❤️ for the trading community**
