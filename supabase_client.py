import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.getenv("https://uhxliyubyjwtpxjcfedk.supabase.co"), os.getenv("sb_publishable_tGsiADOfozJKrf5npbdyew_lS4y4otL"))
