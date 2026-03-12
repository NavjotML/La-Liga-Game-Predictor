import os
from supabase import create_client

SUPABASE_URL=os.environ.get("SUPABASE_URL")
SUPABASE_KEY=os.environ.get("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL,SUPABASE_KEY)

def save_prediction(log_entry:dict):
    supabase.table("predictions").insert(log_entry).execute()

def save_error(error_entry:dict):
    supabase.table("errors").insert(error_entry).execute()

