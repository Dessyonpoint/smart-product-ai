import requests
import json

BASE_URL = "http://localhost:8000"

def test_migrated_api():
    print("🧪 Testing Migrated API...")
    
    # 1. Health check
    try:
        response = requests.get(f"{BASE_URL}/")
        print("✅ API is running:")
        print(json.dumps(response.json(), indent=2))
    except:
        print("❌ API not running. Start with: uvicorn main:app")
        return
    
    # 2. Load your existing data
    print("\n🔄 Loading batch data...")
    try:
        response = requests.post(f"{BASE_URL}/ingest/batch")
        print("✅ Batch data loaded")
    except Exception as e:
        print(f"❌ Batch load failed: {e}")
    
    # 3. Test search with your queries
    test_queries = [
        "wireless audio device",
        "fitness tracking device",
        "computer accessories"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing: '{query}'")
        try:
            response = requests.post(f"{BASE_URL}/search", json={
                "query": query,
                "top_k": 3
            })
            result = response.json()
            print(f"✅ Found {len(result['results'])} results")
            print(f"Method: {result['method_used']}")
            if result['results']:
                top = result['results'][0]
                print(f"Top: {top['title']} (score: {top['similarity_score']:.4f})")
        except Exception as e:
            print(f"❌ Search failed: {e}")
    
    # 4. Get stats
    print(f"\n📊 Database Stats:")
    try:
        response = requests.get(f"{BASE_URL}/stats")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"❌ Stats failed: {e}")

if __name__ == "__main__":
    test_migrated_api()