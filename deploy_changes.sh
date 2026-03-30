#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy_changes.sh  —  Push local changes to GitHub + HuggingFace Space
#
# USAGE:
#   chmod +x deploy_changes.sh          # one-time: make it executable
#   ./deploy_changes.sh "your message"  # commit + push to both remotes
#   ./deploy_changes.sh                 # uses default commit message
#
# WHAT IT DOES (in order):
#   1. Stages all modified tracked files  (git add -u)
#   2. Commits with your message
#   3. Pushes to GitHub  (origin  → github.com/irajkooh/MultiModalRag)
#   4. Pushes to HF Space (space  → huggingface.co/spaces/irajkoohi/MultiModalRag)
#      - fetches LFS objects from space first to avoid non-fast-forward errors
#
# NOTES:
#   - Untracked new files are NOT staged automatically; run `git add <file>` first
#   - If GitHub push fails with "non-fast-forward", run:
#       git pull --rebase origin main && ./deploy_changes.sh "retry"
#   - If Space push fails with "non-fast-forward", run:
#       git pull --rebase space main  && ./deploy_changes.sh "retry"
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MSG="${1:-"chore: update app"}"

echo "▶ Staging modified files..."
git add -u

# Check if there's anything to commit
if git diff --cached --quiet; then
    echo "✅ Nothing to commit — working tree clean."
else
    echo "▶ Committing: \"$MSG\""
    git commit -m "$MSG"
fi

echo "▶ Pushing to GitHub (origin)..."
git push origin main

echo "▶ Fetching LFS objects from HF Space (prevents non-fast-forward errors)..."
git lfs fetch space --all 2>/dev/null || true

echo "▶ Pushing to HuggingFace Space (space)..."
git push space main

echo ""
echo "✅ Deployed successfully!"
echo "   GitHub : https://github.com/irajkooh/MultiModalRag"
echo "   Space  : https://huggingface.co/spaces/irajkoohi/MultiModalRag"
