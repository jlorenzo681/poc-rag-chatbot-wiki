import os
import re

def check_key():
    try:
        with open('.env') as f:
            content = f.read()
            
        # Parse manually to check for common issues like quotes or whitespace being included
        for line in content.splitlines():
            if line.startswith('GROQ_API_KEY='):
                key_value = line.split('=', 1)[1]
                print(f"Key found. Length: {len(key_value)}")
                
                if key_value.startswith('"') or key_value.startswith("'"):
                    print("WARNING: Key starts with quote. Ensure your app handles this or remove quotes in .env if not needed.")
                if key_value.endswith('"') or key_value.endswith("'"):
                    print("WARNING: Key ends with quote.")
                if " " in key_value:
                    print("WARNING: Key contains spaces.")
                
                # Check actual value stripped of potential shell quotes
                clean_key = key_value.strip("'\" ")
                print(f"Cleaned key length: {len(clean_key)}")
                
                if not clean_key.startswith("gsk_"):
                    print("ERROR: Groq API Key usually starts with 'gsk_'")
                else:
                    print("SUCCESS: Key format looks correct (starts with gsk_)")
                    
                return

        print("ERROR: GROQ_API_KEY not found in .env")

    except Exception as e:
        print(f"Error reading .env: {e}")

if __name__ == "__main__":
    check_key()
