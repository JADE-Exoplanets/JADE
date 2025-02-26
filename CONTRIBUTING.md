# Contributing to JADE

Thank you for your interest in contributing to the JADE code! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
  - [Fork and Clone](#fork-and-clone)
  - [Development Environment](#development-environment)
- [Development Workflow](#development-workflow)
  - [Branches](#branches)
  - [Commits](#commits)
  - [Pull Requests](#pull-requests)
- [Coding Standards](#coding-standards)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)
- [Communication](#communication)

## Code of Conduct

Please be respectful, inclusive, and constructive in all interactions.

## Getting Started

### Fork and Clone

1. Fork the repository on GitHub by clicking the "Fork" button at the top-right of the repository page.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/JADE.git
   cd JADE
   ```
3. Add the original repository as an upstream remote:
   ```bash
   git remote add upstream https://github.com/JADE-Exoplanets/JADE.git
   ```

### Development Environment

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Development Workflow

### Branches

- `main`: The primary branch containing stable code
- Create feature branches from `main` with descriptive names:
  ```bash
  git checkout -b feature/new-atmospheric-model
  ```

### Commits

- Write clear, concise commit messages
- Include references to issues if applicable (#issue-number)
- Keep commits focused on single changes when possible

### Pull Requests

1. Keep your fork updated:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. Create your feature branch from the updated main

3. When ready to submit your changes:
   - Push your branch to your fork
   ```bash
   git push origin feature/new-atmospheric-model
   ```
   - Create a pull request from your fork to the main repository
   - Provide a clear title and description
   - Link any related issues

4. Respond to feedback and reviews
   - Make requested changes
   - Push additional commits to your branch (they will automatically appear in the PR)

5. Once approved, a maintainer will merge your changes

## Coding Standards

- Follow PEP 8 style guidelines for Python code
- Use meaningful variable and function names
- Include docstrings for all functions, classes, and modules
- Use type hints where appropriate
- Organize imports alphabetically within their sections

## Documentation

- Update documentation for any changed or new functionality
- Follow the existing documentation style
- Write clear, concise explanations
- Include examples where appropriate

## Issue Reporting

When reporting issues, please include:

1. A clear, descriptive title
2. Steps to reproduce the issue
3. Expected behavior
4. Actual behavior
5. System information (Python version, OS, etc.)
6. Any relevant logs or output

## Communication

- For general issues: Open an issue to discuss
- For security issues: Please report directly to maintainers

Thank you for contributing to JADE!