import os
from dotenv import load_dotenv
from supabase import create_client

print("Step 1: Script started...")

load_dotenv()
print("Step 2: .env loaded...")

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

print(f"Step 3: Keys found? URL: {bool(url)}, Key: {bool(key)}")

supabase = create_client(url, key)
print("Step 4: Client created...")

response = supabase.table("biometric_logs").select("*").limit(1).execute()
print("Step 5: Query finished!")
print(response)