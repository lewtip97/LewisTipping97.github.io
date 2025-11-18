# Render Deployment Guide

## Step 1: Prepare Your Repository

Make sure all files are committed and pushed to GitHub:
- `backend/app.py`
- `backend/requirements.txt`
- `data/prow/ldnpa_prow.gpkg` (the data file must be in your repo)
- `render.yaml` (in the repo root)

## Step 2: Create Render Account

1. Go to [render.com](https://render.com)
2. Sign up or log in (you can use GitHub to sign in)
3. Connect your GitHub account if you haven't already

## Step 3: Create New Web Service

1. Click "New +" in the Render dashboard
2. Select "Web Service"
3. Connect your repository (`LewisTipping97.github.io`)
4. Render should auto-detect the `render.yaml` file

## Step 4: Configure the Service

If auto-detection doesn't work, manually configure:

- **Name**: `wainwright-planner-api` (or your preferred name)
- **Environment**: `Python 3`
- **Build Command**: `pip install -r backend/requirements.txt`
- **Start Command**: `cd backend && uvicorn app:app --host 0.0.0.0 --port $PORT`
- **Plan**: Free tier is fine to start

## Step 5: Deploy

1. Click "Create Web Service"
2. Render will build and deploy your service
3. This may take 5-10 minutes (especially on first deploy)
4. Note your service URL (e.g., `https://wainwright-planner-api.onrender.com`)

## Step 6: Update Frontend

Once deployed, update `wainwrights.html`:

1. Open `wainwrights.html`
2. Find the line: `window.WAINWRIGHT_API_BASE = '';`
3. Replace with: `window.WAINWRIGHT_API_BASE = 'https://your-service-name.onrender.com';`
4. Commit and push to GitHub

## Step 7: Test

1. Visit your GitHub Pages site
2. Navigate to the Wainwright Explorer page
3. Try generating hikes - it should now work!

## Troubleshooting

### Service won't start
- Check the logs in Render dashboard
- Ensure `data/prow/ldnpa_prow.gpkg` is in your repository
- Verify the data file path is correct

### CORS errors
- The backend already has CORS configured to allow all origins
- If issues persist, check the Render service logs

### Slow first request
- Render free tier spins down after inactivity
- First request after spin-down may take 30-60 seconds
- This is normal for free tier

### Data file too large
- If the `.gpkg` file is very large (>100MB), consider:
  - Using Git LFS
  - Or hosting the file separately and downloading on startup

## Free Tier Limitations

- Services spin down after 15 minutes of inactivity
- First request after spin-down is slow (cold start)
- 750 hours/month free (enough for personal projects)

## Upgrading (Optional)

If you want to avoid spin-downs:
- Upgrade to Starter plan ($7/month)
- Service stays always-on
- Faster response times

