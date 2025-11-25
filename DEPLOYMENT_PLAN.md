# Frontend Deployment Plan for iAODE

## Current Branch Structure Analysis

### ✅ **frontend** branch (Source Code)
- **Purpose**: Source code for Next.js projects
- **Contents**: 
  - `datasetsui/` - Dataset browser UI
  - `sc-continuity-explorer/` - Continuity explorer UI
  - All source TypeScript/React files
  - Configuration files (next.config.ts, tsconfig.json, etc.)
  - `node_modules/` (gitignored)

### ⚠️ **gh-pages** branch (Currently Problematic)
- **Current State**: Contains BOTH source code AND node_modules (955MB + 699MB)
- **Problem**: Bloated with dependencies, inefficient for static hosting
- **Should Contain**: ONLY built static files (HTML, CSS, JS, assets)

### **main** branch
- **Purpose**: Backend Python code and API
- **Keep as is**: No frontend code needed here

---

## 🎯 Recommended Solution: Clean Separation

### Option 1: **Two Separate Deployment Paths** (RECOMMENDED)

#### Structure:
```
Branches:
  main           → Backend/API (Python)
  frontend       → Source code for both UIs
  gh-pages       → Built static files ONLY
```

#### Deployment Strategy:

**For datasetsui (Dataset Browser):**
- Build output: Static HTML/CSS/JS
- Deploy to: `https://peterponyu.github.io/iAODE/`
- Path: Root of gh-pages

**For sc-continuity-explorer:**
- Build output: Static HTML/CSS/JS  
- Deploy to: `https://peterponyu.github.io/iAODE/explorer/`
- Path: `/explorer` subdirectory in gh-pages

---

## 📋 Implementation Steps

### Step 1: Configure Static Export

Both projects need `next.config.ts` configured for static export:

**datasetsui/next.config.ts:**
```typescript
const nextConfig = {
  output: 'export',
  basePath: '/iAODE',
  images: {
    unoptimized: true,
  },
  trailingSlash: true,
};
```

**sc-continuity-explorer/next.config.ts:**
```typescript
const nextConfig = {
  output: 'export',
  basePath: '/iAODE/explorer',
  images: {
    unoptimized: true,
  },
  trailingSlash: true,
};
```

### Step 2: Clean Up gh-pages Branch

```bash
# Backup current gh-pages (just in case)
git checkout gh-pages
git branch gh-pages-backup

# Clean gh-pages branch (remove all source code and node_modules)
git checkout --orphan gh-pages-new
git rm -rf .
git clean -fdx

# Add only necessary files for GitHub Pages
echo "# iAODE Frontend" > README.md
git add README.md
git commit -m "Initialize clean gh-pages branch"

# Replace old gh-pages
git branch -D gh-pages
git branch -m gh-pages
git push origin gh-pages --force
```

### Step 3: Set Up GitHub Actions Workflow

Create `.github/workflows/deploy-frontend.yml` in **frontend** branch:

```yaml
name: Deploy Frontend to GitHub Pages

on:
  push:
    branches: [frontend]
    paths:
      - 'datasetsui/**'
      - 'sc-continuity-explorer/**'
      - '.github/workflows/deploy-frontend.yml'
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: true

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout frontend branch
        uses: actions/checkout@v4
        with:
          ref: frontend

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: |
            datasetsui/package-lock.json
            sc-continuity-explorer/package-lock.json

      # Build datasetsui
      - name: Install datasetsui dependencies
        working-directory: ./datasetsui
        run: npm ci

      - name: Build datasetsui
        working-directory: ./datasetsui
        run: npm run build

      # Build sc-continuity-explorer
      - name: Install explorer dependencies
        working-directory: ./sc-continuity-explorer
        run: npm ci

      - name: Build explorer
        working-directory: ./sc-continuity-explorer
        run: npm run build

      # Combine builds
      - name: Prepare deployment directory
        run: |
          mkdir -p deploy
          cp -r datasetsui/out/* deploy/
          mkdir -p deploy/explorer
          cp -r sc-continuity-explorer/out/* deploy/explorer/
          
          # Add .nojekyll to bypass Jekyll processing
          touch deploy/.nojekyll
          
          # Add index redirect if needed
          echo '<!DOCTYPE html>
          <html>
          <head>
            <meta charset="utf-8">
            <title>iAODE</title>
            <meta http-equiv="refresh" content="0; url=./datasets/">
          </head>
          <body>
            <p>Redirecting to <a href="./datasets/">datasets browser</a>...</p>
          </body>
          </html>' > deploy/index.html

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: ./deploy

      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

### Step 4: Enable GitHub Pages

1. Go to repository Settings → Pages
2. Source: **GitHub Actions** (not "Deploy from branch")
3. This allows the workflow to deploy directly

---

## 🔒 Safety Assessment

### Current gh-pages Branch Issues:
- ❌ **Contains source code**: Unnecessary duplication
- ❌ **Contains node_modules**: 1.6GB of dependencies
- ❌ **Mixed with backend files**: Python scripts, JSON data
- ❌ **Build artifacts (.next)**: Should only have final output

### After Cleanup:
- ✅ **Only static files**: HTML, CSS, JS, assets
- ✅ **Small size**: ~10-50MB (vs 1.6GB+)
- ✅ **Fast deployment**: No build needed on gh-pages
- ✅ **Clean separation**: Source in frontend, output in gh-pages
- ✅ **Automated**: GitHub Actions handles everything

---

## 🚀 Deployment Workflow

```
Developer → Commit to 'frontend' branch
              ↓
        GitHub Actions triggered
              ↓
        1. Checkout frontend
        2. Install dependencies
        3. Build datasetsui → out/
        4. Build explorer → out/
        5. Combine outputs
        6. Deploy to gh-pages branch
              ↓
        GitHub Pages serves:
        - https://peterponyu.github.io/iAODE/
        - https://peterponyu.github.io/iAODE/explorer/
```

---

## 📊 File Structure After Deployment

```
gh-pages branch (Clean):
├── .nojekyll
├── index.html (redirect)
├── datasets/
│   ├── index.html
│   ├── _next/
│   │   └── static/...
│   └── ...
├── explorer/
│   ├── index.html
│   ├── _next/
│   │   └── static/...
│   ├── data/
│   │   ├── chunks/
│   │   └── metadata/
│   └── ...
└── statistics/
    └── index.html

frontend branch (Source):
├── datasetsui/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── next.config.ts
├── sc-continuity-explorer/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── next.config.ts
└── .github/
    └── workflows/
        └── deploy-frontend.yml
```

---

## ⚡ Quick Start Commands

### Test Local Build:
```bash
cd datasetsui
npm run build
npx serve out

cd ../sc-continuity-explorer
npm run build
npx serve out
```

### Manual Deployment (if needed):
```bash
# Build both projects
cd datasetsui && npm run build
cd ../sc-continuity-explorer && npm run build

# Prepare gh-pages
git checkout gh-pages
git pull origin gh-pages

# Clear old content
rm -rf *

# Copy new builds
cp -r ../frontend/datasetsui/out/* .
mkdir -p explorer
cp -r ../frontend/sc-continuity-explorer/out/* explorer/

# Commit and push
git add .
git commit -m "Deploy: $(date +'%Y-%m-%d %H:%M')"
git push origin gh-pages
```

---

## 🎯 Next Steps

1. **Commit all build fixes** to frontend branch
2. **Create GitHub Actions workflow** file
3. **Clean gh-pages branch** (remove source code and node_modules)
4. **Push frontend branch** to trigger first automated deployment
5. **Verify deployment** at GitHub Pages URL
6. **Test both UIs** in production

---

## 📝 Notes

- **No source code in gh-pages**: Keeps it clean and secure
- **Automated deploys**: Push to frontend = auto-deploy
- **Version control**: Source code stays in git history
- **Rollback capability**: Can revert frontend commits if needed
- **Fast hosting**: GitHub Pages only serves static files efficiently
