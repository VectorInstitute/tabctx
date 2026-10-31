#!/usr/bin/env bash
# Build the hosted agent demo with Cloud Build and deploy it to Cloud Run.
#
#   examples/agent_demo/deploy.sh                       # project coderd, us-central1
#   GCP_ACCOUNT=you@vectorinstitute.ai examples/agent_demo/deploy.sh
#
# The service is public (anyone can replay the showcase runs); starting a run
# needs a Vector gateway key, checked by the app. Sessions live in memory, so
# the service runs one instance; it scales to zero when idle, and sign-ins
# end when it does. Nothing secret goes into the image or the service.
set -euo pipefail

PROJECT=${GCP_PROJECT:-coderd}
REGION=${GCP_REGION:-us-central1}
SERVICE=${SERVICE:-tabctx-agent-demo}
REPOSITORY=${REPOSITORY:-tabctx-agent-demo}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TAG=$(git -C "$ROOT" rev-parse --short HEAD)$(git -C "$ROOT" diff --quiet || echo "-dirty")
IMAGE=$REGION-docker.pkg.dev/$PROJECT/$REPOSITORY/demo:$TAG
gcloud=(gcloud --project "$PROJECT" ${GCP_ACCOUNT:+--account "$GCP_ACCOUNT"})

if ! "${gcloud[@]}" artifacts repositories describe "$REPOSITORY" --location "$REGION" >/dev/null 2>&1; then
  "${gcloud[@]}" artifacts repositories create "$REPOSITORY" --location "$REGION" \
    --repository-format docker --description "tabctx agent demo images"
fi

"${gcloud[@]}" builds submit "$ROOT" --region "$REGION" \
  --config "$ROOT/examples/agent_demo/cloudbuild.yaml" --substitutions "_IMAGE=$IMAGE"

# One instance (in-memory sessions); room for 3 agent runs at once; up to an
# hour per request for the live event streams; CPU always on, so a run keeps
# going when its viewer closes the tab.
"${gcloud[@]}" run deploy "$SERVICE" --region "$REGION" --image "$IMAGE" \
  --allow-unauthenticated --port 8080 \
  --min-instances 0 --max-instances 1 --concurrency 80 \
  --cpu 4 --memory 8Gi --no-cpu-throttling --execution-environment gen2 \
  --timeout 3600 \
  --set-env-vars DEMO_MAX_RUNS=3,DEMO_RUN_TIMEOUT=900 \
  --labels app=tabctx-agent-demo

"${gcloud[@]}" run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)'
