"""Quick endpoint test for all fixed features"""
import urllib.request
import json
import time

BASE = "http://localhost:5000"

def test_endpoint(name, url, method="GET", data=None, expect_html=False):
    print(f"\n{'='*50}")
    print(f"Testing: {name}")
    print(f"URL: {url}")
    try:
        if data:
            req = urllib.request.Request(
                url, 
                data=json.dumps(data).encode(),
                headers={"Content-Type": "application/json"}
            )
        else:
            req = urllib.request.Request(url)
        
        resp = urllib.request.urlopen(req, timeout=120)
        
        if expect_html:
            html = resp.read().decode()
            print(f"Status: {resp.status}")
            print(f"  HTML length: {len(html)} chars")
            # Check for key content
            checks = {
                "CineSense": "CineSense" in html,
                "Chat AI nav": "Chat AI" in html or "chat-ui" in html,
                "Friends nav": "Friends" in html or "friends" in html,
            }
            for check, passed in checks.items():
                print(f"  {check}: {'YES' if passed else 'NO'}")
            print("PASS")
        else:
            body = json.loads(resp.read())
            print(f"Status: {resp.status}")
            
            # Print summary
            if isinstance(body, dict):
                for k, v in body.items():
                    if isinstance(v, list):
                        print(f"  {k}: [{len(v)} items]")
                        if v and isinstance(v[0], dict):
                            print(f"    First: {v[0].get('title', v[0])}")
                    elif isinstance(v, str) and len(v) > 100:
                        print(f"  {k}: {v[:100]}...")
                    else:
                        print(f"  {k}: {v}")
            print("PASS")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"HTTP Error: {e.code}")
        print(f"  Body: {body[:300]}")
        print("FAIL")
    except Exception as e:
        print(f"Error: {e}")
        print("FAIL")

# Test 1: Home page loads
test_endpoint("Home Page", f"{BASE}/", expect_html=True)

# Test 2: Trending
test_endpoint("Trending API", f"{BASE}/api/trending")

# Test 3: Chat
test_endpoint("Chat API", f"{BASE}/api/chat", data={"message": "recommend me a good sci-fi movie"})

# Test 4: Mood Recommendations
test_endpoint("Mood API", f"{BASE}/api/mood-recommendations", data={"mood": "happy", "user_id": 1})

# Test 5: Chat UI page
test_endpoint("Chat UI Page", f"{BASE}/chat-ui", expect_html=True)

# Test 6: Friends page (will redirect to login)
test_endpoint("Friends/Login Page", f"{BASE}/friends", expect_html=True)

# Test 7: Features page
test_endpoint("Features Page", f"{BASE}/features", expect_html=True)

print(f"\n{'='*50}")
print("All tests complete!")
