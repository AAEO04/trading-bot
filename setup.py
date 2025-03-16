import json
import hashlib
import getpass

def initial_setup():
    print("=== Trading Bot Setup ===")
    
    config = {
        "telegram_token": input("Enter Telegram bot token: ").strip(),
        "password_hash": hashlib.sha256(
            getpass.getpass("Set password (min 8 chars): ").encode()
        ).hexdigest()
    }
    with open("config.json", "w") as f:
        json.dump(config, f)
        
    print("Setup complete! Run 'python bot.py' to start.")

if __name__ == "__main__":
    initial_setup()