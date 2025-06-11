import os

class Settings:
    def __init__(self):
        # Default to the deployed BentoML endpoint
        self.api_url = os.getenv("API_URL", "https://rag-service-complete-ebc82948.mt-guc1.bentoml.ai")
        self.bentoml_endpoint = os.getenv("BENTOML_ENDPOINT", "https://rag-service-complete-ebc82948.mt-guc1.bentoml.ai")
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.s3_bucket_name = os.getenv("S3_BUCKET_NAME", "bentoml-rag-storage-ayush-2024")
        self.dynamodb_table_name = os.getenv("DYNAMODB_TABLE_NAME", "rag-document-metadata")

settings = Settings()