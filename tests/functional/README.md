# Functional Tests for ragctl

## Vue d'Ensemble

Cette suite de tests fonctionnels crée un **filet de sécurité** avant de nettoyer le code mort. Elle valide que toutes les fonctionnalités de ragctl continuent de fonctionner correctement.

## Stratégie

1. **Documenter** toutes les fonctionnalités (48 test cases)
2. **Créer** des données de test
3. **Exécuter** la suite de tests → baseline
4. **Nettoyer** le code mort par petites étapes
5. **Re-exécuter** les tests après chaque étape
6. **Revert** si des tests échouent

## Fichiers

- `FUNCTIONAL_TESTS.md` - Documentation exhaustive des 48 test cases
- `setup_test_data.sh` - Génère les données de test
- `test_ragctl.sh` - Exécute automatiquement tous les tests
- `README.md` - Ce fichier

## Usage

### Étape 1: Setup (Première fois)

```bash
# Générer les données de test
./tests/functional/setup_test_data.sh
```

**Sortie attendue:**
```
🔧 Setting up test data for functional tests...
📄 Creating test.txt...
📄 Creating empty.txt...
📄 Creating large.txt...
...
✅ Test data setup complete!
```

**Fichiers créés:**
- `test_data/test.txt` - Texte simple (~1KB)
- `test_data/empty.txt` - Fichier vide
- `test_data/large.txt` - Texte volumineux (~100KB)
- `test_data/chunks.json` - JSON valide
- `test_data/chunks.jsonl` - JSONL valide
- `test_data/invalid.json` - JSON malformé
- `test_data/docs/` - Répertoire avec fichiers mixtes
- `test_data/empty/` - Répertoire vide
- `test_data/mixed/` - Fichiers de types différents
- `test_data/test.pdf` - PDF (si pdflatex installé)

### Étape 2: Baseline (Avant Cleanup)

```bash
# Exécuter tous les tests
./tests/functional/test_ragctl.sh
```

**Sortie attendue:**
```
🚀 Starting ragctl Functional Tests
Test Data: ./test_data
Output: ./test_output
Log: ./test_output/test_results.log

═══════════════════════════════════════════════════════
 1. Testing: ragctl chunk
═══════════════════════════════════════════════════════
✅ PASS: 1.1 Chunk simple text file
✅ PASS: 1.2 Chunk with output path
✅ PASS: 1.3 Chunk with strategy semantic
...

═══════════════════════════════════════════════════════
 Test Summary
═══════════════════════════════════════════════════════

Total Tests:   48
Passed:        42
Failed:        0
Skipped:       6

🎉 All tests passed!
```

**Note**: Certains tests peuvent être skippés si:
- PDF non généré (pas de pdflatex)
- Qdrant non démarré (tests ingest)
- Pas de runs failed (tests retry)

### Étape 3: Pendant le Cleanup

Après chaque suppression de code:

```bash
# Re-exécuter les tests
./tests/functional/test_ragctl.sh

# Si des tests échouent
git revert HEAD

# Si tout passe
git commit -m "cleanup: removed dead code - all tests pass"
```

## Structure des Tests

### 1. ragctl chunk (14 tests)
- ✅ Fichier texte simple
- ✅ Différentes stratégies (semantic, token, sentence)
- ✅ Options (max-tokens, overlap, show)
- ✅ PDF et OCR avancé
- ✅ Gestion d'erreurs
- ✅ Fichiers volumineux

### 2. ragctl batch (10 tests)
- ✅ Traitement de répertoires
- ✅ Patterns (*.txt, *.pdf, *.md)
- ✅ Mode récursif
- ✅ Auto-continue sur erreurs
- ✅ Types de fichiers mixtes

### 3. ragctl ingest (9 tests)
- ⚠️ Requiert Qdrant running
- ✅ Ingestion JSON/JSONL
- ✅ Collections personnalisées
- ✅ URLs personnalisées

### 4. ragctl eval (7 tests)
- ✅ Évaluation multi-stratégies
- ✅ Comparaison de stratégies
- ✅ PDF support
- ✅ Fichiers volumineux

### 5. ragctl info (3 tests)
- ✅ Affichage informations système
- ✅ API URL personnalisée
- ✅ Fonctionne sans API

### 6. ragctl retry (5 tests)
- ✅ Afficher runs échoués
- ⚠️ Requiert des runs échoués pour tests complets

## Logs et Outputs

Tous les outputs sont dans `test_output/`:
- `test_results.log` - Log principal
- `test_X_Y.log` - Log de chaque test individuel
- `chunks_X_Y.json` - Outputs des tests chunk
- `batch_X_Y/` - Outputs des tests batch

## Dépannage

### Problème: "Test data not found"
**Solution**: Exécuter `./tests/functional/setup_test_data.sh`

### Problème: Tous les tests ingest skipped
**Solution**: Démarrer Qdrant
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Problème: Tests PDF skipped
**Solution**: Installer texlive ou créer test.pdf manuellement
```bash
# macOS
brew install --cask mactex-no-gui

# Ubuntu
sudo apt-get install texlive-latex-base

# Ou créer PDF manuellement
cp mon_document.pdf test_data/test.pdf
```

### Problème: Permission denied
**Solution**: Rendre les scripts exécutables
```bash
chmod +x tests/functional/*.sh
```

## Interpréter les Résultats

### ✅ PASS
Test réussi - fonctionnalité OK

### ❌ FAIL
Test échoué - **ATTENTION**: Revert le dernier commit !
```bash
git revert HEAD
```

### ⚠️ SKIP
Test ignoré - dépendance manquante (Qdrant, PDF, etc.)
- C'est OK si skippé avant et après cleanup
- **ATTENTION** si skippé après cleanup mais pas avant !

## Workflow Recommandé

### Phase 1: Baseline
```bash
# 1. Générer test data
./tests/functional/setup_test_data.sh

# 2. Exécuter tests baseline
./tests/functional/test_ragctl.sh > baseline_results.txt

# 3. Vérifier résultats
cat baseline_results.txt
# Note: X tests passed, Y skipped

# 4. Commit baseline
git add test_data/ baseline_results.txt
git commit -m "test: add functional test baseline"
```

### Phase 2: Cleanup Incrémental
```bash
# 1. Identifier fichier à supprimer (DEAD_CODE_ANALYSIS.md)
# Exemple: langchain_loader_old.py

# 2. Supprimer le fichier
git rm src/workflows/ingest/langchain_loader_old.py

# 3. Tester IMMÉDIATEMENT
./tests/functional/test_ragctl.sh

# 4a. Si PASS → Commit
git commit -m "cleanup: remove langchain_loader_old.py - dead code"

# 4b. Si FAIL → Revert
git checkout src/workflows/ingest/langchain_loader_old.py
# Analyser pourquoi ça échoue
```

### Phase 3: Validation Finale
```bash
# Après plusieurs cleanups
./tests/functional/test_ragctl.sh > final_results.txt

# Comparer avec baseline
diff baseline_results.txt final_results.txt

# Si identiques (sauf noms de fichiers) → SUCCESS
```

## Métriques de Succès

| Métrique | Baseline | Après Cleanup | Status |
|----------|----------|---------------|--------|
| Tests PASS | 42 | 42 | ✅ OK |
| Tests FAIL | 0 | 0 | ✅ OK |
| Tests SKIP | 6 | 6 | ✅ OK |
| **Total** | **48** | **48** | **✅ OK** |

## Questions Fréquentes

**Q: Combien de temps prennent les tests ?**
A: ~2-5 minutes (sans ingest/PDF)

**Q: Dois-je tous les lancer à chaque fois ?**
A: Oui ! C'est rapide et ça garantit qu'on ne casse rien.

**Q: Que faire si un test échoue ?**
A:
1. Lire `test_output/test_X_Y.log`
2. Identifier le code supprimé responsable
3. Revert: `git revert HEAD`
4. Marquer le code comme "utilisé" dans DEAD_CODE_ANALYSIS.md

**Q: Puis-je ajouter mes propres tests ?**
A: Oui ! Éditer `test_ragctl.sh` et ajouter un test case.

**Q: Les tests modifient-ils mes données ?**
A: Non, ils utilisent uniquement `test_data/` et `test_output/`.

## Prochaines Étapes

Une fois les tests en place:

1. ✅ Exécuter baseline → documenter résultats
2. ⬜ Commencer cleanup (catégorie SAFE d'abord)
3. ⬜ Tests après chaque étape
4. ⬜ Documenter progrès cleanup
5. ⬜ Mesurer impact sur coverage

---

**Créé**: 2025-10-29
**Version**: 1.0
**Auteur**: Claude Code
**Objectif**: Nettoyer le code mort en toute sécurité
