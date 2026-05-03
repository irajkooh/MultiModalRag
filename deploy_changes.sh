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
#   4. Pushes to HF Space via a clean orphan branch — binary data files
#      (PDF, PNG, DOCX) are excluded from the Space push because HF Space
#      does not support Git LFS; those files live in the HF Dataset repo
#      irajkoohi/MultiModalRag_dataset and are downloaded at Space startup.
#
# DATA FILES (persistent across Space restarts):
#   - Add/remove files in data/ and run:
#       python3 -c "
#       from huggingface_hub import HfApi
#       import os, sys
#       api = HfApi(token=os.environ['HF_TOKEN'])
#       api.upload_file(path_or_fileobj=sys.argv[1],
#                       path_in_repo='data/'+os.path.basename(sys.argv[1]),
#                       repo_id='irajkoohi/MultiModalRag_dataset',
#                       repo_type='dataset')
#       " data/yourfile.pdf
#
# NOTES:
#   - Untracked new files are NOT staged automatically; run `git add <file>` first
#   - If GitHub push fails with "non-fast-forward", run:
#       git pull --rebase origin main && ./deploy_changes.sh "retry"
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MSG="${1:-"chore: update app"}"
BINARY_EXTS=("*.pdf" "*.png" "*.jpg" "*.jpeg" "*.docx" "*.xlsx")

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

# ── HF Space push via clean orphan branch (no binary files) ──────────────────
echo "▶ Building clean Space deploy branch (binary files excluded)..."
git checkout --orphan space-deploy
git add -A

# Remove binary data files from the index (they live in HF Dataset instead)
for ext in "${BINARY_EXTS[@]}"; do
    # shellcheck disable=SC2046
    git rm --cached $(git ls-files "data/$ext") 2>/dev/null || true
done

git commit -m "$MSG [space deploy]"

echo "▶ Force-pushing to HuggingFace Space..."
git push space space-deploy:main --force

echo "▶ Returning to main branch..."
git checkout -f main
git branch -D space-deploy

echo ""
echo "✅ Deployed successfully!"
echo "   GitHub : https://github.com/irajkooh/MultiModalRag"
echo "   Space  : https://huggingface.co/spaces/irajkoohi/MultiModalRag"
echo "   Dataset: https://huggingface.co/datasets/irajkoohi/MultiModalRag_dataset"
