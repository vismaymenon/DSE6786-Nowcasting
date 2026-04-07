import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.getenv("PROJECT_URL"), os.getenv("PROJECT_KEY"))
