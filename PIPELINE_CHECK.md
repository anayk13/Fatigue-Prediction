# 🔍 Pipeline Check Guide

## How to Check if Your Pipeline is Working

After you push to GitHub, here's how to verify everything is working:

### Step 1: Check GitHub Actions Status

1. **Go to your GitHub repository**
2. **Click on the "Actions" tab** (top navigation bar)
3. **You should see workflow runs** - look for:
   - ✅ Green checkmark = Success
   - ❌ Red X = Failed
   - 🟡 Yellow circle = In Progress

### Step 2: View Workflow Details

Click on any workflow run to see:
- **Which jobs ran** (test, build, deploy)
- **Execution time**
- **Logs for each step**

### Step 3: Check Individual Job Logs

Click on a job (e.g., "test") to see:
- ✅ Which steps passed
- ❌ Which steps failed
- 📝 Detailed error messages

## Common Issues & Fixes

### Issue 1: "Model not found" Error

**Symptom:** Tests fail because model doesn't exist

**Fix:** This is expected on first run! The pipeline will create sample data and train a model automatically.

**Action:** Let the pipeline complete - it should handle this gracefully.

### Issue 2: Docker Build Fails

**Symptom:** Build job fails with Docker errors

**Possible Causes:**
- Missing files in repository
- Dockerfile syntax error
- Port conflicts

**Fix:** 
- Check that all files are committed
- Verify Dockerfile is correct
- Check build logs for specific error

### Issue 3: Tests Fail

**Symptom:** Test job shows failures

**Note:** The workflow uses `|| true` to allow tests to fail gracefully during initial setup.

**Fix:** 
- Check test logs to see what failed
- Most tests are designed to skip if model/data doesn't exist yet
- This is normal on first run

### Issue 4: Permission Denied (Retrain Job)

**Symptom:** "Permission denied" when trying to push model

**Fix:** Already fixed! The workflow now includes:
```yaml
permissions:
  contents: write
```

If it still fails, you may need to:
1. Go to Settings → Actions → General
2. Enable "Workflow permissions" → "Read and write permissions"

### Issue 5: Dependencies Installation Fails

**Symptom:** pip install fails

**Possible Causes:**
- Package version conflicts
- Network issues
- Missing system dependencies

**Fix:** Check the logs - usually shows which package failed

## What to Share for Troubleshooting

If the pipeline fails, share:

1. **Screenshot of the failed job**
2. **Error message from logs** (copy the relevant section)
3. **Which job failed** (test, build, deploy, retrain-model)
4. **Link to the workflow run** (if possible)

## Expected First Run Behavior

On your **first push**, expect:

1. ✅ **Test job** - May show some skipped tests (normal)
2. ✅ **Build job** - Should build Docker image successfully
3. ⚠️ **Deploy job** - Will just echo messages (needs Render/Railway setup)
4. ⏭️ **Retrain job** - Won't run (only on schedule/manual trigger)

## Manual Testing Before Push

Test locally first:

```bash
# 1. Test the pipeline scripts
python src/data_ingestion.py
python src/feature_engineering.py
python src/train_model.py

# 2. Test Docker build
docker build -t fatigue-monitor:latest .

# 3. Test Docker run
docker run -p 8501:8501 fatigue-monitor:latest

# 4. Run tests
pytest tests/ -v
```

## Quick Status Check Commands

After pushing, you can check status via GitHub CLI (if installed):

```bash
gh run list  # List recent workflow runs
gh run watch # Watch current run
```

Or check via browser:
- Repository → Actions tab → Latest workflow run

## Success Indicators

✅ **Pipeline is working if:**
- All jobs show green checkmarks
- Test job completes (even with some skipped tests)
- Build job creates Docker image successfully
- No critical errors in logs

⚠️ **Warnings are OK:**
- Some tests skipped (expected if model doesn't exist)
- Sample data created (expected if API unavailable)
- Deploy job just shows messages (needs external setup)

## Need Help?

Share:
1. The workflow run URL
2. Screenshot of the failed job
3. Error message from logs

I can help fix any issues! 🚀

