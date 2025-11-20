# 🤝 Contributing to SmartCommerce-AI

Thank you for your interest in contributing to SmartCommerce-AI! This document provides guidelines for contributing to this project.

## 🌟 Ways to Contribute

### 🐛 **Bug Reports**
- Use the GitHub Issues template for bug reports
- Include detailed steps to reproduce
- Provide system information (OS, Python version, etc.)
- Include screenshots if applicable

### 💡 **Feature Requests** 
- Describe the feature and its use case
- Explain how it would benefit users
- Consider implementation complexity
- Discuss potential alternatives

### 🔧 **Code Contributions**
- Fix bugs or implement new features
- Improve documentation
- Add test cases
- Optimize performance

### 📚 **Documentation**
- Improve README or guides
- Add code comments
- Create tutorials or examples
- Update API documentation

## 🚀 Development Setup

### **1. Fork and Clone**
```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/SmartCommerce-AI.git
cd SmartCommerce-AI
```

### **2. Set Up Development Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install pytest black flake8 mypy
```

### **3. Initialize Database**
```bash
python -c "from data.database import init_database; init_database()"
```

## 📝 Development Guidelines

### **Code Style**
- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and small
- Use type hints where appropriate

### **Example Code Style**
```python
def predict_product_price(
    product_features: Dict[str, Any],
    model_type: str = "random_forest"
) -> Tuple[float, float]:
    """
    Predict product price using specified ML model.
    
    Args:
        product_features: Dictionary containing product attributes
        model_type: Type of ML model to use for prediction
        
    Returns:
        Tuple of (predicted_price, confidence_score)
    """
    # Implementation here
    pass
```

### **Project Structure**
- **Models**: Add new ML models in `models/` directory
- **Services**: External integrations go in `services/`
- **UI Components**: New pages in `pages/` directory
- **Utilities**: Helper functions in `utils/`
- **Configuration**: Settings in `config/`

### **Database Changes**
- Modify schema in `data/database.py`
- Provide migration scripts if needed
- Update sample data generation accordingly
- Test with both SQLite and production databases

## 🧪 Testing

### **Running Tests**
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_price_predictor.py

# Run with coverage
pytest --cov=models --cov-report=html
```

### **Writing Tests**
- Add tests for new features in `tests/` directory
- Test both happy path and edge cases
- Mock external dependencies (SerpAPI, database)
- Aim for >80% code coverage

### **Example Test**
```python
def test_price_prediction():
    """Test price prediction functionality."""
    predictor = PricePredictor()
    
    # Test data
    product_data = {
        'category': 'Electronics',
        'brand': 'Apple',
        'rating': 4.5,
        'reviews_count': 1000
    }
    
    # Test prediction
    price, confidence = predictor.predict(product_data)
    
    assert isinstance(price, float)
    assert 0 <= confidence <= 1
    assert price > 0
```

## 📋 Pull Request Process

### **1. Create Feature Branch**
```bash
git checkout -b feature/amazing-new-feature
```

### **2. Make Changes**
- Write clean, well-documented code
- Add or update tests as needed
- Update documentation if required
- Follow the coding standards

### **3. Test Changes**
```bash
# Run tests
pytest

# Check code style
black .
flake8 .

# Type checking
mypy .
```

### **4. Commit Changes**
```bash
git add .
git commit -m "Add amazing new feature

- Implement feature X with Y algorithm
- Add comprehensive tests
- Update documentation
- Resolves #123"
```

### **5. Push and Create PR**
```bash
git push origin feature/amazing-new-feature
```

Then create a Pull Request on GitHub with:
- Clear title and description
- Reference related issues
- Screenshots if UI changes
- Checklist of changes made

## 🏷️ Issue Labels

- **bug**: Something isn't working correctly
- **enhancement**: New feature or improvement
- **documentation**: Documentation needs improvement
- **good first issue**: Good for newcomers
- **help wanted**: Extra attention needed
- **priority: high**: Critical issues
- **priority: low**: Nice to have features

## 🎯 Areas Needing Contribution

### **High Priority**
- 🔧 **Performance Optimization**: Improve prediction speed
- 🧪 **Test Coverage**: Add more comprehensive tests
- 📱 **Mobile UI**: Improve mobile responsiveness
- 🔐 **Security**: Enhance data protection and API security

### **Medium Priority**
- 🌐 **Additional Data Sources**: Integrate more price comparison sites
- 🤖 **Advanced ML**: Implement deep learning models
- 📊 **Analytics**: Add more business intelligence features
- 🐳 **Docker Support**: Create production Docker containers

### **Good First Issues**
- 📚 **Documentation**: Improve code comments and guides
- 🎨 **UI Improvements**: Enhance visual design
- 🐛 **Bug Fixes**: Resolve reported issues
- 🧪 **Test Cases**: Add tests for existing functionality

## 📞 Getting Help

### **Community**
- 💬 **GitHub Discussions**: Ask questions and share ideas
- 🐛 **GitHub Issues**: Report bugs or request features
- 📧 **Email**: Contact maintainers directly

### **Development Questions**
- How to add a new ML model?
- How to integrate a new data source?
- How to modify the database schema?
- How to add a new UI component?

## 🏆 Recognition

Contributors will be:
- Listed in the project README
- Mentioned in release notes
- Given credit in relevant documentation
- Invited to be project maintainers (for significant contributions)

## 📜 Code of Conduct

### **Our Standards**
- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers learn and contribute
- Maintain professional communication

### **Unacceptable Behavior**
- Harassment or discrimination
- Inappropriate language or content
- Spam or self-promotion
- Disruptive behavior

## 🎉 Thank You!

Every contribution, no matter how small, makes SmartCommerce-AI better for everyone. We appreciate your time and effort in improving this project!

---

**Happy Coding! 🚀**