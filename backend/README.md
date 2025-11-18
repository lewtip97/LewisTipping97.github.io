# Wainwright Planner API

FastAPI backend for generating hiking routes through Wainwright peaks.

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the data file exists at `../data/prow/ldnpa_prow.gpkg`

3. Run the server:
```bash
uvicorn app:app --host localhost --port 8005
```

## Render Deployment

The app is configured to deploy on Render. The `render.yaml` file in the repo root contains the deployment configuration.

### Setup on Render:

1. Connect your GitHub repository to Render
2. Render will automatically detect the `render.yaml` file
3. The service will build and deploy automatically
4. Note your service URL (e.g., `https://wainwright-planner-api.onrender.com`)

### Environment Variables:

No additional environment variables are required. The PORT is automatically set by Render.

### Data File:

Make sure `data/prow/ldnpa_prow.gpkg` is committed to your repository and accessible from the backend directory.

