
gcloud ml-engine models create trajectory --regions us-central1
gcloud ml-engine models list
gcloud ml-engine versions list --model trajectory
gcloud ml-engine versions create v1 --model trajectory --origin $MODEL_BINARY --runtime-version 1.4
gcloud ml-engine predict --model trajectory --version v1 --json-instances batch_predict.json


