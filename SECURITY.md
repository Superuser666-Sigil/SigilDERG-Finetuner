# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.8.x   | :white_check_mark: |
| < 2.8   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

1. **DO NOT** open a public GitHub issue for security vulnerabilities
2. Email security concerns to the project maintainers
3. Include as much detail as possible:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: Within 48 hours of your report
- **Initial Assessment**: Within 1 week
- **Fix Timeline**: Depends on severity, typically 1-4 weeks
- **Disclosure**: Coordinated disclosure after fix is released

## Security Scope

### In Scope

- Code execution vulnerabilities in evaluation sandbox
- Data leakage through model outputs
- Dependency vulnerabilities in core packages
- Authentication/authorization issues
- Path traversal or injection attacks

### Out of Scope

- Vulnerabilities in upstream dependencies (report to those projects)
- Denial of service through resource exhaustion
- Social engineering attacks
- Physical security

## Security Measures

### Code Evaluation Sandbox

LLM-generated code is evaluated in isolated environments:

1. **Docker Isolation** (Recommended)
   - Network disabled
   - Filesystem read-only except temp directories
   - Resource limits enforced
   - User namespace isolation

2. **Firejail Sandbox** (Linux)
   - Seccomp filters
   - Capability dropping
   - Private mount namespace

3. **No Sandbox** (Development Only)
   - Only for trusted code
   - Not recommended for production

### Dependency Security

- Dependencies are pinned to specific versions
- Regular updates via automated PR checks
- Security scanning with `pip-audit` in CI

### Model Security

- Model weights are loaded from trusted sources (HuggingFace Hub)
- Gated models require authentication
- No arbitrary code execution from model outputs

### Data Security

- Training data is processed locally
- No automatic data upload to external services
- Logs do not contain sensitive information

## Best Practices

### For Users

1. **Use Sandboxed Evaluation**: Always use Docker or Firejail for untrusted code
2. **Review Generated Code**: Never execute generated code without review
3. **Keep Dependencies Updated**: Run `pip install --upgrade` regularly
4. **Use Gated Models**: Authenticate for gated model access

### For Contributors

1. **No Secrets in Code**: Use environment variables for credentials
2. **Validate Inputs**: All user inputs must be validated
3. **Sandbox Untrusted Code**: Any code execution must be sandboxed
4. **Review Dependencies**: Check new dependencies for security issues

## Known Limitations

1. **Sandbox Escape**: While rare, sandbox escapes are possible. Use defense in depth.
2. **Model Hallucinations**: LLMs may generate insecure code patterns.
3. **Resource Exhaustion**: Large models require significant resources; DoS is possible.

## Security Updates

Security updates are released as patch versions (e.g., 2.8.1, 2.8.2). Subscribe to releases to stay informed.

## Acknowledgments

We thank the security researchers who have helped improve our security posture.

