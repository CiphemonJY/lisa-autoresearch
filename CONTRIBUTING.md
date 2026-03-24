# Contributing to LISA + AutoResearch

Thank you for your interest in contributing! This project aims to democratize AI by enabling large model training on consumer hardware.

## 🌟 Key Contribution Areas

### High Priority
1. **Performance Optimization**
   - Async I/O for disk operations
   - Activation compression
   - Mixed precision support

2. **Platform Support**
   - Windows native support
   - GPU acceleration (CUDA/Metal)
   - Multi-GPU training

3. **Documentation**
   - Usage examples
   - Performance benchmarks
   - Model compatibility

### Welcome Contributions
- Bug fixes
- Documentation improvements
- Test cases
- Performance benchmarks
- New model support

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- MLX (macOS) or PyTorch (Linux/Windows)
- 16GB+ RAM for 32B training

### Development Setup

```bash
# Clone the repository
git clone https://github.com/CiphemonJY/LISA_FTM.git
cd LISA_FTM

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black

# Run tests
pytest tests/

# Format code
black .
```

## 📝 Code Style

- Follow PEP 8
- Use Black for formatting
- Add docstrings to functions
- Keep functions focused and testable

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_disk_offload.py

# Run with coverage
pytest --cov=lisa_autoresearch tests/
```

## 📚 Documentation

- Update README.md for user-facing changes
- Update CHANGELOG.md for all changes
- Add inline comments for complex logic
- Update docstrings for modified functions

## 🔧 Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### PR Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] PR description clear

## 💬 Communication

- **Issues**: Bug reports, feature requests, questions
- **Discussions**: General questions, ideas, show & tell
- **Pull Requests**: Code contributions

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in:
- README.md (contributors section)
- Release notes
- Project documentation

---

## Project Structure

```
LISA_FTM/
├── LISA_FTM/              # Main package
│   ├── __init__.py
│   ├── disk_offload.py          # Disk-offload training
│   └── train.py                 # Training entry point
├── disk_offload.py              # Implementation
├── train_qwen7b.py              # Training script
├── test_32b_training.py         # Test suite
├── tests/                       # Unit tests
├── docs/                        # Documentation
├── examples/                    # Usage examples
├── README.md                    # Main documentation
├── CHANGELOG.md                 # Version history
├── CONTRIBUTING.md              # This file
├── LICENSE                      # MIT License
└── requirements.txt             # Dependencies
```

## Need Help?

- Check [README.md](README.md) for usage
- Open an [Issue](https://github.com/CiphemonJY/LISA_FTM/issues) for bugs
- Start a [Discussion](https://github.com/CiphemonJY/LISA_FTM/discussions) for questions

Thank you for helping democratize AI training! 🎉���