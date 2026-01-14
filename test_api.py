"""Quick test to check API response"""
import requests

response = requests.get('http://localhost:5000/api/compare')
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
