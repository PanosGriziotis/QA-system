import os
import six
from google.cloud import translate_v2 as translate
from google.cloud import translate
import google.cloud.storage as gcs
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './googlekey.json'

class GoogleApi():
    def __init__(self) -> None:
        self.client = gcs.Client()
        self.source_bucket_name = 'source_en_bucket'
        self.target_bucket_name = 'target_el_bucket'

    def batch_translate_text(self,
    filename,
    save_dir,
    project_id:str="qa-subsytem",
    ):
        input_uri= f"gs://{self.source_bucket_name}/{filename}"
        output_uri= f"gs://{self.target_bucket_name}/{save_dir}"

        client = translate.TranslationServiceClient()
        location = "us-central1"
        gcs_source = {"input_uri": input_uri}

        input_configs_element = {
        "gcs_source": gcs_source,
        "mime_type": "text/plain", 
        }
        gcs_destination = {"output_uri_prefix": output_uri}
        output_config = {"gcs_destination": gcs_destination}
        parent = f"projects/{project_id}/locations/{location}"

        operation = client.batch_translate_text(
        request={
            "parent": parent,
            "source_language_code": "en",
            "target_language_codes": ["el"],  # Up to 10 language codes here.
            "input_configs": [input_configs_element],
            "output_config": output_config,
        }
        )

        print("Waiting for translation operation to complete...")
        response = operation.result()
        print("Total Characters: {}".format(response.total_characters))
        print("Translated Characters: {}".format(response.translated_characters))

    def upload_file_to_bucket (self, input_filepath):
        """upload file for translation to gcs source bucket"""
        filename = os.path.basename(input_filepath)
        bucket = self.client.get_bucket(self.source_bucket_name)
        bucket.blob(filename).upload_from_filename(input_filepath)
        print (f"{filename} uploaded in {self.source_bucket_name}: DONE")

    def download_file (self, filename):
        bucket = self.client.get_bucket(self.source_bucket_name)
        blob =  bucket.get_blob(filename)
        blob.download_to_filename(filename)
        
    def download_translated_file (self,translation_filename):
        
        """download translated data from gcs target bucket"""
        bucket = self.client.get_bucket(self.target_bucket_name)
        blobs = [blob.name for blob in bucket.list_blobs()]
        translated_file = list(filter(lambda x:x.endswith('.txt'), blobs))[0]
        blob =  bucket.get_blob(translated_file)
        blob.download_to_filename(translation_filename)
        

# source_en_bucket_.._who_queries.en_el_translations.txt

#upload_sourcefile_to_bucket('./test_file.txt')
#batch_translate_text()
