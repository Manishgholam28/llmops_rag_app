from langfuse import get_client
from dotenv import load_dotenv

#load the api keys
load_dotenv()  

langfuse = get_client()

langfuse.update_prompt(
    name = "rag_app_system_prompt",
    version = None,
    new_labels = ["staging"]
)