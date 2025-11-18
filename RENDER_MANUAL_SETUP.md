# Manual Render Setup (If render.yaml doesn't work)

If you're getting errors with the `render.yaml` file, set up the service manually in Render's dashboard:

## Step-by-Step Manual Setup

1. **Go to Render Dashboard** → Click "New +" → "Web Service"

2. **Connect Repository**
   - Select your GitHub repository: `LewisTipping97.github.io`
   - Click "Connect"

3. **Configure Service Settings**

   **Basic Settings:**
   - **Name**: `wainwright-planner-api`
   - **Region**: Choose closest to you
   - **Branch**: `dev` (or `main` if you merged)
   - **Root Directory**: `backend` ⚠️ **IMPORTANT: Set this to `backend`**
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

   **Plan:**
   - Select "Free" plan

4. **Environment Variables**
   - No additional environment variables needed
   - `PORT` is automatically set by Render

5. **Click "Create Web Service"**

6. **Wait for Deployment**
   - First build takes 5-10 minutes
   - Watch the logs for any errors

## Common Issues

### Issue: "Couldn't find package.json"
**Solution**: Make sure "Root Directory" is set to `backend` in Render dashboard

### Issue: "Module not found"
**Solution**: Check that Root Directory is `backend` and requirements.txt is in that directory

### Issue: "Data file not found"
**Solution**: Ensure the `data/` directory is committed to your repo and accessible from the backend directory

## After Deployment

Once deployed, you'll get a URL like: `https://wainwright-planner-api.onrender.com`

Update `wainwrights.html` line 648:
```javascript
window.WAINWRIGHT_API_BASE = 'https://wainwright-planner-api.onrender.com';
```

Then commit and push to GitHub.

