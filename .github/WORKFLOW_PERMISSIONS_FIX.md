# Fix: GitHub Actions Permissions for SARIF Upload

**Date**: 2025-10-29
**Issue**: CodeQL SARIF upload failing with "Resource not accessible by integration"
**Status**: ✅ Fixed

---

## 🔍 Problem

The `trivy-scan` job was failing to upload SARIF results to GitHub Code Scanning with the error:

```
Error: Resource not accessible by integration
Warning: This run of the CodeQL Action does not have permission to access
the CodeQL Action API endpoints.
```

### Root Causes

1. **Missing permissions**: The `trivy-scan` job didn't have `security-events: write` permission
2. **Fork PRs**: Pull requests from forks don't have access to upload to Code Scanning

---

## ✅ Solution Applied

### 1. Added Required Permissions

**File**: `.github/workflows/pr-validation.yml`
**Lines**: 263-265

```yaml
trivy-scan:
  name: Trivy Vulnerability Scan
  runs-on: ubuntu-latest
  permissions:
    contents: read
    security-events: write  # ✅ Added - Required for uploading SARIF
  steps:
    # ...
```

### 2. Added Fork Detection

**Lines**: 278-290

```yaml
- name: Upload Trivy results to CodeQL
  uses: github/codeql-action/upload-sarif@v3
  # Skip upload for forks (no permissions) - keep local artifact
  if: always() && github.event.pull_request.head.repo.full_name == github.repository
  with:
    sarif_file: 'trivy-results.sarif'

- name: Upload Trivy results as artifact (for forks)
  uses: actions/upload-artifact@v4
  if: always()
  with:
    name: trivy-results
    path: trivy-results.sarif
```

### Benefits

- ✅ **CodeQL upload works** for internal PRs and branches
- ✅ **Forks don't fail** - results uploaded as artifacts instead
- ✅ **Results preserved** in both cases (CodeQL or artifacts)
- ✅ **No breaking changes** for contributors

---

## 📋 Permissions Reference

### Security-Related Jobs Permissions

| Job | Permissions | Why |
|-----|-------------|-----|
| `codeql` | `security-events: write` | Upload CodeQL analysis results |
| `trivy-scan` | `security-events: write` | Upload Trivy SARIF results |
| `supply-chain` | `contents: read` | Read repo for SBOM generation |

### Typical Workflow Permissions

```yaml
permissions:
  contents: read          # Read repo content
  security-events: write  # Upload security scan results
  actions: read           # Read workflow status
  pull-requests: write    # Comment on PRs (optional)
```

---

## 🧪 Testing

### Verify the Fix

1. **Push to branch**:
   ```bash
   git add .github/workflows/pr-validation.yml
   git commit -m "fix: add security-events permission to trivy-scan"
   git push origin test/workflow-permissions
   ```

2. **Check workflow run**:
   - Go to Actions tab in GitHub
   - Find the `PR Validation` workflow
   - Check `trivy-scan` job - should succeed ✅

3. **Verify upload**:
   - Go to Security tab → Code scanning alerts
   - Should see Trivy results uploaded

### For Fork PRs

1. **Contributor opens PR from fork**
2. **Workflow runs**:
   - CodeQL upload: Skipped (no permissions)
   - Artifact upload: ✅ Success
3. **Maintainer can download artifact** to review results

---

## 📖 Related Documentation

- [GitHub Actions Permissions](https://docs.github.com/en/actions/security-guides/automatic-token-authentication#permissions-for-the-github_token)
- [SARIF Upload](https://docs.github.com/en/code-security/code-scanning/integrating-with-code-scanning/uploading-a-sarif-file-to-github)
- [CodeQL Action](https://github.com/github/codeql-action)
- [Trivy Action](https://github.com/aquasecurity/trivy-action)

---

## 🔒 Security Considerations

### Why `security-events: write` is Safe

- ✅ **Limited scope**: Only allows uploading security scan results
- ✅ **No code access**: Cannot modify code or workflows
- ✅ **Read-only repo**: `contents: read` prevents repo modifications
- ✅ **Standard practice**: Required for all SARIF uploads

### Fork Security

The fork detection ensures that:
- Forks cannot upload to main repo's Code Scanning
- Results are still preserved as downloadable artifacts
- Maintainers can review before merging

---

## 🚀 Other Jobs with SARIF Upload

If you add more security scanning tools that generate SARIF, remember to add permissions:

```yaml
your-security-job:
  name: Your Security Scan
  runs-on: ubuntu-latest
  permissions:
    contents: read
    security-events: write  # ✅ Required for SARIF upload
  steps:
    - name: Run scan
      # ...

    - name: Upload results
      uses: github/codeql-action/upload-sarif@v3
      if: always() && github.event.pull_request.head.repo.full_name == github.repository
      with:
        sarif_file: 'results.sarif'
```

---

## ✅ Checklist

- [x] Added `security-events: write` permission to `trivy-scan`
- [x] Added fork detection for CodeQL upload
- [x] Added artifact upload fallback for forks
- [x] Tested on internal branch ✅
- [ ] Tested on fork PR (optional)

---

## 📊 Before/After Comparison

### Before (❌ Failing)

```yaml
trivy-scan:
  name: Trivy Vulnerability Scan
  runs-on: ubuntu-latest
  # ❌ No permissions specified
  steps:
    - name: Upload Trivy results
      uses: github/codeql-action/upload-sarif@v3
      if: always()  # ❌ Always tries, fails on forks
```

**Result**: ❌ Error on all PRs (including forks)

### After (✅ Working)

```yaml
trivy-scan:
  name: Trivy Vulnerability Scan
  runs-on: ubuntu-latest
  permissions:
    contents: read
    security-events: write  # ✅ Added
  steps:
    - name: Upload Trivy results to CodeQL
      uses: github/codeql-action/upload-sarif@v3
      if: always() && github.event.pull_request.head.repo.full_name == github.repository  # ✅ Fork check

    - name: Upload as artifact (fallback)
      uses: actions/upload-artifact@v4
      if: always()  # ✅ Always works
```

**Result**: ✅ Works for internal PRs, graceful fallback for forks

---

**Author**: Claude AI Security Audit
**Last Updated**: 2025-10-29
**Status**: ✅ Resolved
