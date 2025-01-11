import subprocess
import requests
import os

GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN') 
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable is not set")
GITHUB_API_URL = "https://api.github.com"
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

# Fetch VM IP using hostname -i
def get_vm_ip():
    try:
        result = subprocess.run(["curl", "ifconfig.me"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        vm_ip = result.stdout.strip()
        vm_ip = vm_ip.split(" ")[0]  # Extract the first IP address 
        return vm_ip
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to fetch VM IP using hostname -i: {e.stderr.strip()}")

# Generate webhook URL
def get_webhook_url():
    vm_ip = get_vm_ip()
    return f"http://{vm_ip}:8000/webhook"
    # return "https://just_added.com/webhook"

print(get_vm_ip())
print(get_webhook_url())

# # Repositories to which the webhook will be added
REPOSITORIES = [
    {"owner": "Unstudio-Tech", "repo": "ai-toolkit"},
]

def create_webhook(owner, repo, webhook_url):
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/hooks"
    payload = {
        "name": "web",
        "active": True,
        "events": ["push", "pull_request"],
        "config": {
            "url": webhook_url,
            "content_type": "json"
        }
    }
    
    response = requests.post(url, headers=HEADERS, json=payload)
    print(response.json())
    if response.status_code == 201:
        print(f"Webhook successfully created for {owner}/{repo}")
    else:
        print(HEADERS)
        print(f"Failed to create webhook for {owner}/{repo}: {response.json()}")

# Main logic
if __name__ == "__main__":
    try:
        webhook_url = get_webhook_url()
        print(f"Webhook URL: {webhook_url}")
        for repo in REPOSITORIES:
            create_webhook(repo["owner"], repo["repo"], webhook_url)
    except Exception as e:
        print(f"Error: {e}")