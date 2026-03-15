#!/usr/bin/env bash
set -euo pipefail

# Rollback Deployment - TradeMaster
# Usage: ./rollback-deploy.sh [commits_back]

COMMITS_BACK="${1:-1}"

echo "=== TradeMaster Deployment Rollback ==="
echo "Rolling back $COMMITS_BACK commit(s)"
echo "Time: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo ""

# Step 1: Check current state
echo "[1/5] Current state:"
echo "  Branch: $(git branch --show-current)"
echo "  Commit: $(git log --oneline -1)"
echo ""

# Step 2: Verify target commit
TARGET=$(git log --oneline -$((COMMITS_BACK + 1)) | tail -1)
echo "[2/5] Target rollback commit: $TARGET"
read -p "  Proceed? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Step 3: Create rollback branch
ROLLBACK_BRANCH="rollback/$(date +%Y%m%d-%H%M%S)"
echo "[3/5] Creating rollback branch: $ROLLBACK_BRANCH"
git checkout -b "$ROLLBACK_BRANCH"
git revert --no-commit HEAD~${COMMITS_BACK}..HEAD
git commit -m "rollback: revert last $COMMITS_BACK commit(s)"

# Step 4: Push and deploy
echo "[4/5] Pushing rollback branch..."
git push origin "$ROLLBACK_BRANCH"

# Step 5: Verify
echo "[5/5] Rollback branch pushed. Create PR to merge into main."
echo ""
echo "=== Rollback Complete ==="
echo "Next steps:"
echo "  1. Create PR from $ROLLBACK_BRANCH to main"
echo "  2. Merge and verify deployment"
echo "  3. Check health: curl <BACKEND_URL>/api/v1/system/health"
