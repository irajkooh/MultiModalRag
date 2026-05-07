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
RESET_DB=false
for arg in "$@"; do [[ "$arg" == "--reset-db" ]] && RESET_DB=true; done

if $RESET_DB; then
    echo "▶ Clearing stale vectorstore from HF Hub dataset..."
    python3 - <<'PYEOF'
import os, sys, re
token = os.environ.get("MultiModalRag_Token", "").strip()
if not token:
    # Try loading from _secrets/HF_TOKEN.txt — extract the hf_... token line
    try:
        with open("_secrets/HF_TOKEN.txt") as f:
            for line in f:
                line = line.strip()
                if re.match(r'^hf_[A-Za-z0-9]+$', line):
                    token = line
                    break
    except Exception:
        pass
if not token:
    print("⚠  HF token not found — skipping DB reset")
    sys.exit(0)
from huggingface_hub import HfApi
api = HfApi(token=token)
repo = "irajkoohi/MultiModalRag_dataset"
try:
    files = [f for f in api.list_repo_files(repo, repo_type="dataset") if f.startswith("vectorstore/")]
    for f in files:
        api.delete_file(path_in_repo=f, repo_id=repo, repo_type="dataset",
                        commit_message="reset vectorstore")
    print(f"✅  Cleared {len(files)} vectorstore file(s) from HF Hub dataset")
except Exception as e:
    print(f"⚠  DB reset failed: {e}")
PYEOF
fi

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

# ── HF Space push via a temp directory (never touches working tree) ──────────
echo "▶ Building clean Space deploy branch (binary files excluded)..."

_tmpdir=$(mktemp -d)
# Copy entire working tree to temp dir, excluding what doesn't belong on Space
rsync -a --exclude='.git' \
         --exclude='data/*.pdf' \
         --exclude='data/*.png' \
         --exclude='data/*.jpg' \
         --exclude='data/*.jpeg' \
         --exclude='data/*.docx' \
         --exclude='data/*.xlsx' \
         --exclude='data/images/' \
         --exclude='data/tables/' \
         --exclude='vectorstore/' \
         --exclude='vectorstore_corrupted_backup/' \
         --exclude='_secrets/' \
         --exclude='.venv/' \
         --exclude='__pycache__/' \
         --exclude='*.pyc' \
         . "$_tmpdir/"

# Build an orphan git repo in the temp dir and push it
pushd "$_tmpdir" > /dev/null
git init -q
git checkout -b space-deploy
git add -A
git commit -q -m "$MSG [space deploy]"
echo "▶ Force-pushing to HuggingFace Space..."
git remote add space "$(cd - > /dev/null && git remote get-url space)"
git push space space-deploy:main --force
popd > /dev/null
rm -rf "$_tmpdir"

echo ""
echo "✅ Deployed successfully!"
echo "   GitHub : https://github.com/irajkooh/MultiModalRag"
echo "   Space  : https://huggingface.co/spaces/irajkoohi/MultiModalRag"
echo "   Dataset: https://huggingface.co/datasets/irajkoohi/MultiModalRag_dataset"
