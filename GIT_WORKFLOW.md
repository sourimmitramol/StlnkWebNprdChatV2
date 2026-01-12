# Git Workflow Guide: Merging and Reverting

This guide outlines the steps to merge your current changes in `dev_fixer` into the `dev` (development) and `main` (production) branches, as well as how to revert changes if necessary.

## Part 1: Merging Changes Forward

You are currently on the **`dev_fixer`** branch.

### Step 1: Commit Your Changes
First, ensure all your current work is saved.

```bash
# Check status
git status

# Add changed files (ensure you only add what you want)
git add . 

# Commit the changes
git commit -m "Fix python environment and langchain dependencies"

# Push changes to remote dev_fixer branch
git push origin dev_fixer
```

### Step 2: Merge into `dev` (Development Branch)
Once your fixes are secured in `dev_fixer`, merge them into the `dev` branch for integration testing.

```bash
# Switch to the dev branch
git checkout dev

# Pull the latest changes to ensure you are up to date
git pull origin dev

# Merge your fix branch into dev
git merge dev_fixer

# Push updated dev branch to remote
git push origin dev
```

### Step 3: Merge into `main` (Production Branch)
After validating `dev`, merge changes into `main`.

```bash
# Switch to the main branch
git checkout main

# Pull latest main to avoid conflicts
git pull origin main

# Merge dev (which now contains your fixes) into main
git merge dev

# Push updated main branch to remote
git push origin main
```

---

## Part 2: Moving Backward (Reverting Changes)

If something breaks or you need to undo changes, here are the strategies based on how far you've gone.

### Scenario A: You haven't committed yet
If you made changes to files but haven't run `git commit`.

```bash
# Discard changes in a specific file
git checkout -- <filename>

# Discard ALL changes in the current directory (Be careful!)
git checkout .
```

### Scenario B: You comitted, but haven't pushed yet
If you made a commit on your local branch but haven't pushed it to the server.

```bash
# Undo the last commit, but keep your file changes (soft reset)
git reset --soft HEAD~1

# Undo the last commit and DESTROY file changes (hard reset)
git reset --hard HEAD~1
```

### Scenario C: You passed changes (Pushed) and need to revert
If changes are already on `remote` (GitHub/GitLab/etc.), **do not use reset**. Instead, create a new commit that does the opposite of the bad commit.

```bash
# 1. Find the commit ID you want to undo
git log --oneline

# 2. Revert that specific commit (git will create a new commit for this)
git revert <commit-id>

# 3. Push the reversion
git push origin <branch-name>
```

### Scenario D: Emergency - Reset Branch to match Remote
If your local branch is messy and you just want it to match exactly what is safely on the server.

```bash
# Fetch latest from server
git fetch origin

# Force local branch to match remote branch (Destroys local unpushed work)
git reset --hard origin/main
```
