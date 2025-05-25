# 🚀 AI Stock Chart Assistant - Quick Start Guide

## ✅ Installation Complete!

Your AI Stock Chart Assistant is now ready to use. All dependencies are installed and the application has been tested successfully.

## 🔑 Next Steps

### 1. Get Your Google Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the API key

### 2. Configure Your API Key
1. Open the `.env` file in your project root
2. Replace `your_gemini_api_key_here` with your actual API key:
   ```
   GEMINI_API_KEY=AIzaSyB7M51Kxb3EK-avdlwVPJPATqpErKcKp14
   ```
3. Save the file

### 3. Run the Application
```bash
python main.py
```

Or use the launcher scripts:
- **Windows**: Double-click `run_app.bat`
- **Linux/Mac**: Run `./run_app.sh`

## 🎯 How to Use

1. **Upload Image**: Click "📁 Select Image" or drag & drop a stock chart
2. **Analyze**: Click "🤖 Analyze Chart" to get AI analysis
3. **Review Results**: View technical analysis in the center panel
4. **Save/Export**: Save to history or export as PDF/Text/JSON

## 📊 Supported Chart Types

- Candlestick charts
- Line charts
- Bar charts
- Technical analysis charts
- Any stock chart image (PNG, JPG, JPEG, GIF, BMP)

## ⌨️ Keyboard Shortcuts

- `Ctrl+O`: Open image file
- `Ctrl+S`: Save session
- `F5`: Refresh analysis
- `Ctrl+Q`: Quit application

## 🛠️ Features Available

✅ **AI-Powered Analysis** - Google Gemini Vision API integration
✅ **Professional GUI** - Modern three-column layout
✅ **Error Handling** - Comprehensive error management
✅ **Session Recovery** - Auto-save and restore
✅ **Analysis History** - Keep track of all analyses
✅ **Multiple Export Formats** - PDF, Text, JSON
✅ **Real-time Status** - Progress indicators and health monitoring
✅ **Theme Support** - Dark, light, and system themes

## 🔧 Troubleshooting

**If the application doesn't start:**
1. Make sure you have a valid Gemini API key in `.env`
2. Check that all dependencies are installed: `pip install -r requirements_new.txt`
3. Run the test script: `python test_installation.py`

**For analysis errors:**
1. Ensure your image is a clear stock chart
2. Check internet connection
3. Verify API key is valid and has quota remaining

## 📞 Need Help?

- Check the full `README.md` for detailed documentation
- Review logs in the `data/logs/` directory
- Use the "Test API Connection" button in the app settings

---

**🎉 Enjoy analyzing your stock charts with AI!**
