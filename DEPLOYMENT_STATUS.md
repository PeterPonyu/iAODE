# ✅ Deployment Setup Complete

## Summary

Successfully configured both frontend projects for GitHub Pages deployment with automated CI/CD.

---

## 🎯 Branch Strategy (SAFE & CLEAN)

### ✅ **frontend** branch
- **Purpose**: Source code repository
- **Contains**: 
  - `datasetsui/` source code
  - `sc-continuity-explorer/` source code  
  - GitHub Actions workflow
  - Configuration files
- **Size**: ~50MB (no node_modules in git)
- **Status**: ✅ Ready for development

### ⚠️ **gh-pages** branch (NEEDS CLEANUP)
- **Current Status**: Contains source code + 1.6GB node_modules ❌
- **Should Contain**: Only static build outputs (~10-20MB) ✅
- **Action Required**: Clean up before first automated deployment

### ✅ **main** branch
- **Purpose**: Backend Python code
- **Status**: Independent, no changes needed

---

## 📦 Build Configuration

### datasetsui - ✅ WORKING
**Configuration** (`datasetsui/next.config.ts`):
```typescript
{
  output: 'export',
  basePath: '/iAODE',
  images: { unoptimized: true },
  trailingSlash: true,
}
```

**Build Status**: ✅ Success
- Generated: `out/` directory (6.2MB)
- Pages: 120 static pages
- URL: `https://peterponyu.github.io/iAODE/`

### sc-continuity-explorer - ✅ WORKING  
**Configuration** (`sc-continuity-explorer/next.config.ts`):
```typescript
{
  output: 'export',
  basePath: '/iAODE/explorer',
  images: { unoptimized: true },
  trailingSlash: true,
}
```

**Build Status**: ✅ Success (verified earlier)
- URL: `https://peterponyu.github.io/iAODE/explorer/`

---

## 🚀 GitHub Actions Workflow

**File**: `.github/workflows/deploy-frontend.yml`
**Status**: ✅ Created and ready

**Trigger**: 
- Push to `frontend` branch
- Manual dispatch

**Process**:
1. Checkout frontend branch
2. Build datasetsui → `out/`
3. Build sc-continuity-explorer → `out/`
4. Combine builds into `deploy/` directory
5. Deploy to GitHub Pages

---

## 🔧 Fixes Applied

### datasetsui
1. ✅ Fixed TypeScript errors in `dataLoader.ts` (H5AnalysisData interface)
2. ✅ Fixed ESLint errors (unused variables, imports)
3. ✅ Fixed `getCategoryInfo` return type (added `icon` property)
4. ✅ Fixed static export issues:
   - Created `GSEDetailWrapper` for client-side searchParams
   - Created `DatasetBrowserWrapper` for client-side type selection
   - Removed `searchParams` from server components

### sc-continuity-explorer
1. ✅ Fixed Plotly type error (`easing: 'cubic-in-out' as const`)
2. ✅ Added `eslint-disable` comments for necessary `any` types
3. ✅ Configured for static export

---

## 📋 Next Steps to Complete Deployment

### Step 1: Commit All Changes
```bash
cd /home/zeyufu/Desktop/LAB/iAODE_dev

# Add all changes
git add -A

# Commit
git commit -m "Configure static export and fix build errors

- Add GitHub Actions workflow for automated deployment
- Configure both UIs for static export with proper base paths
- Fix TypeScript and ESLint errors
- Create client wrappers to handle searchParams in static export
- Add deployment documentation"

# Push to frontend branch
git push origin frontend
```

### Step 2: Clean gh-pages Branch (IMPORTANT)
```bash
# Create backup first
git checkout gh-pages
git branch gh-pages-backup-$(date +%Y%m%d)

# Option A: Clean manually (keep data files)
git checkout gh-pages
rm -rf datasetsui/node_modules datasetsui/.next datasetsui/src
rm -rf sc-continuity-explorer/node_modules sc-continuity-explorer/.next sc-continuity-explorer/src
git add -A
git commit -m "Remove source code and node_modules from gh-pages"
git push origin gh-pages

# Option B: Start fresh (recommended)
git checkout --orphan gh-pages-new
git rm -rf .
echo "# iAODE - Deployed via GitHub Actions" > README.md
git add README.md  
git commit -m "Initialize clean gh-pages"
git branch -D gh-pages
git branch -m gh-pages
git push origin gh-pages --force
```

### Step 3: Enable GitHub Pages
1. Go to: https://github.com/PeterPonyu/iAODE/settings/pages
2. **Source**: Select "GitHub Actions"
3. Save

### Step 4: Trigger First Deployment
```bash
# Push to frontend branch (if not already done)
git checkout frontend
git push origin frontend

# Or manually trigger workflow
# Go to: Actions → Deploy Frontend to GitHub Pages → Run workflow
```

### Step 5: Verify Deployment
After workflow completes (~2-3 minutes):
- Main UI: https://peterponyu.github.io/iAODE/
- Datasets: https://peterponyu.github.io/iAODE/datasets/
- Statistics: https://peterponyu.github.io/iAODE/statistics/
- Explorer: https://peterponyu.github.io/iAODE/explorer/

---

## 🔒 Safety Analysis

### ✅ Safe to Remove from gh-pages:
- ✅ `datasetsui/src/` - Source in frontend branch
- ✅ `datasetsui/node_modules/` - 699MB, rebuilt on each deploy  
- ✅ `sc-continuity-explorer/src/` - Source in frontend branch
- ✅ `sc-continuity-explorer/node_modules/` - 955MB, rebuilt on each deploy
- ✅ `datasetsui/.next/` - Build cache, regenerated
- ✅ `sc-continuity-explorer/.next/` - Build cache, regenerated
- ✅ Python files (*.py) - Should be in main branch only
- ✅ API files - Backend belongs in main branch

### ⚠️ Keep in gh-pages (after deployment):
- Data files only if needed for explorer
- Static assets from `public/` directories
- Generated HTML/CSS/JS from builds

### 🎯 Ideal gh-pages size: ~10-50MB
### ❌ Current gh-pages size: ~1.6GB

---

## 📊 File Structure (After Full Setup)

```
Repository:
├── main branch
│   ├── api/
│   ├── iaode/
│   ├── *.py (Python backend)
│   └── README.md
│
├── frontend branch
│   ├── .github/workflows/deploy-frontend.yml
│   ├── datasetsui/
│   │   ├── src/
│   │   ├── public/
│   │   ├── package.json
│   │   └── next.config.ts
│   ├── sc-continuity-explorer/
│   │   ├── src/
│   │   ├── public/
│   │   ├── package.json
│   │   └── next.config.ts
│   └── DEPLOYMENT_PLAN.md
│
└── gh-pages branch (CLEAN)
    ├── .nojekyll
    ├── index.html
    ├── datasets/
    │   └── index.html
    ├── statistics/
    │   └── index.html
    ├── explorer/
    │   └── index.html
    └── _next/
        └── static/
```

---

## 🎉 Benefits

1. **Automated Deployment**: Push to frontend = automatic deploy
2. **Clean Separation**: Source vs. output
3. **Version Control**: Full history in git
4. **Fast Hosting**: Only static files served
5. **Easy Rollback**: Revert commits if needed
6. **No Manual Work**: GitHub Actions handles everything
7. **Small Deploy Size**: ~10-50MB vs. 1.6GB

---

## 🐛 Troubleshooting

### If build fails in GitHub Actions:
1. Check Actions tab for error logs
2. Test locally: `npm run build` in each directory
3. Verify `next.config.ts` settings
4. Check Node.js version (should be 20)

### If pages don't load:
1. Verify basePath in next.config.ts
2. Check GitHub Pages settings
3. Look for 404 errors in browser console
4. Ensure .nojekyll file exists

### If data doesn't load in explorer:
1. Check public/data/ directory in build output
2. Verify data paths in code
3. Check browser console for fetch errors

---

## 📝 Maintenance

### To update UIs:
1. Make changes in `frontend` branch
2. Test locally: `npm run build`
3. Commit and push
4. GitHub Actions auto-deploys

### To rollback:
```bash
git checkout frontend
git revert <commit-hash>
git push origin frontend
```

---

**Status**: ✅ Ready for deployment
**Next**: Commit changes and push to trigger first automated build
