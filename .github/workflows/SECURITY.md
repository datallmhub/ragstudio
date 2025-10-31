# 🔒 Security Tools for AI/ML Projects

This project implements an **enterprise-grade security stack** specifically designed for AI/ML applications.

---

## 📋 Integrated Security Tools

### 1. **Secrets Detection** 🔑

**Tools:**
- **Gitleaks** - Detects secrets in code (API keys, tokens, passwords)
- **TruffleHog** - Scans for verified secrets

**Detects:**
- ✅ OpenAI, Anthropic, HuggingFace API keys
- ✅ AWS/GCP/Azure credentials
- ✅ Database passwords
- ✅ Private keys
- ✅ Tokens in Git history

**When:** On every PR

---

### 2. **SAST (Static Application Security Testing)** 🔍

**Tools:**
- **Bandit** - Python security linter
- **Semgrep** - Pattern matching with AI/ML rules

**Detects:**
- ✅ SQL injection
- ✅ Hardcoded secrets
- ✅ Unsafe deserialization (pickle files)
- ✅ Path traversal
- ✅ YAML/JSON injection
- ✅ Unsafe file operations

**AI-Specific Rules:**
- Loading unverified models
- Dangerous ML data deserialization
- Dynamic code execution

**When:** On every PR

---

### 3. **Dependency Security** 📦

**Tools:**
- **Safety** - PyPI vulnerability scanner
- **Pip-audit** - Alternative dependency checker
- **Snyk** - Comprehensive CVE database

**Checks:**
- ✅ torch, transformers, tensorflow
- ✅ langchain, openai, anthropic
- ✅ numpy, scipy, pandas
- ✅ All transitive dependencies

**Why Critical for AI:**
- ML libraries often have critical vulnerabilities
- Supply chain attacks on pre-trained models
- Backdoors in model weights

**When:** On every PR + release

---

### 4. **Supply Chain Security** 🔗

**Tools:**
- **Dependency Review** - GitHub native
- **SBOM Generation** - Software Bill of Materials (CycloneDX)

**Generates:**
- ✅ Complete dependency list
- ✅ Exact versions
- ✅ Licenses
- ✅ Verification hashes

**Use Cases:**
- Full traceability
- Compliance auditing
- Tampering detection

**When:** On every PR + release

---

### 5. **License Compliance** ⚖️

**Tools:**
- **pip-licenses** - License extraction

**Checks:**
- ❌ Blocks GPL, AGPL, LGPL (copyleft)
- ✅ Allows MIT, Apache, BSD

**Why Critical for AI:**
- Many ML models have restrictive licenses
- HuggingFace models may be non-commercial
- Prevents legal issues

**When:** On every PR

---

### 6. **CodeQL (Advanced SAST)** 🧠

**Tool:**
- **GitHub CodeQL** - Semantic code analysis

**Analyzes:**
- ✅ Data flow analysis
- ✅ Taint tracking
- ✅ Control flow analysis
- ✅ Security patterns

**Queries:**
- `security-extended` - Extended vulnerabilities
- `security-and-quality` - Quality + security

**When:** On every PR

---

### 7. **Trivy (Vulnerability Scanner)** 🛡️

**Tool:**
- **Aqua Trivy** - Universal scanner

**Scans:**
- ✅ Filesystem
- ✅ Python dependencies
- ✅ OS packages
- ✅ Containers (if applicable)

**Severity:**
- CRITICAL, HIGH, MEDIUM

**When:** On every PR + release

---

## 🚨 AI/ML Specific Use Cases

### Risk 1: Model Poisoning
**Tool:** Bandit + Semgrep
**Detects:** Loading unverified models
```python
# ❌ Dangerous
model = torch.load("model.pth")  # Detected by Bandit

# ✅ Safe
model = torch.load("model.pth", map_location="cpu", weights_only=True)
```

### Risk 2: Data Exfiltration
**Tool:** Semgrep + CodeQL
**Detects:** Sending sensitive data
```python
# ❌ Dangerous - Detected
requests.post(UNKNOWN_URL, data=user_data)

# ✅ Safe - Whitelisted URLs only
```

### Risk 3: Pickle Deserialization
**Tool:** Bandit
**Detects:** Unsafe pickle usage
```python
# ❌ Dangerous
import pickle
data = pickle.load(file)  # Detected

# ✅ Safe
import json
data = json.load(file)
```

### Risk 4: API Key Leaks
**Tool:** Gitleaks + TruffleHog
**Detects:** Tokens in code
```python
# ❌ Dangerous - Detected immediately
OPENAI_API_KEY = "sk-proj-abc123..."

# ✅ Safe
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

---

## 🔧 Required Configuration

### GitHub Secrets (Optional)

**SNYK_TOKEN** (recommended):
1. Create account at https://snyk.io
2. Generate API token
3. Add to GitHub Secrets

**Without Snyk token:**
All other tools work without configuration!

---

## 📊 Generated Reports

Each PR generates:
- 📄 Bandit JSON report
- 📄 Semgrep JSON report
- 📄 Safety JSON report
- 📄 Pip-audit JSON report
- 📄 SBOM (CycloneDX JSON)
- 📄 Licenses JSON/Markdown
- 📄 Trivy SARIF

**Access:**
GitHub Actions → Artifacts

---

## ✅ Best Practices

### For Contributors:

1. **Before committing:**
   ```bash
   # Local scan
   gitleaks detect --source .
   bandit -r src/
   ```

2. **Test dependencies:**
   ```bash
   safety check
   pip-audit
   ```

3. **Check licenses:**
   ```bash
   pip-licenses --fail-on="GPL;AGPL"
   ```

### For Admin:

1. **Review Security tab** on GitHub regularly
2. **Check Dependabot alerts**
3. **Audit SBOM** before each release
4. **Validate licenses** of new dependencies

---

## 🆘 Alert Response

### CRITICAL Vulnerability Found:

1. **Do not merge PR**
2. **Identify dependency:** Check report
3. **Find patch:**
   ```bash
   pip install --upgrade <package>
   ```
4. **If no patch:** Find alternative

### Secret Detected:

1. **STOP immediately**
2. **Revoke secret** (OpenAI, AWS, etc.)
3. **Remove from history:**
   ```bash
   git filter-branch --force --index-filter \
     'git rm --cached --ignore-unmatch <file>' HEAD
   ```
4. **Force new secret**

### Incompatible License:

1. **Identify dependency**
2. **Find alternative** with compatible license
3. **Or negotiate** commercial license

---

## 📚 Resources

- [OWASP Top 10 for LLM](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [HuggingFace Model Cards](https://huggingface.co/docs/hub/model-cards)
- [Microsoft AI Security Best Practices](https://www.microsoft.com/en-us/security/business/ai-machine-learning)

---

## 🎯 Summary

| Risk | Tool | When |
|------|------|------|
| Secrets in code | Gitleaks, TruffleHog | Every PR |
| Code vulnerabilities | Bandit, Semgrep, CodeQL | Every PR |
| Dependency CVEs | Safety, Pip-audit, Snyk | Every PR |
| Supply chain | SBOM, Dependency Review | Every PR |
| License issues | pip-licenses | Every PR |
| Container vulns | Trivy | Every PR + Release |

**Result:** Enterprise-grade security stack for AI projects! 🛡️
