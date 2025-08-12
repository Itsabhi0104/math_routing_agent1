import numpy as np
import lancedb

# Simple test script to verify the search works
print("Testing LanceDB search...")

try:
    # Connect to the database
    conn = lancedb.connect("lancedb_math")
    
    # Open the table
    tbl = conn.open_table("math_qa")
    
    # Create a dummy query vector (384 dimensions)
    query_vector = np.random.random(384).tolist()
    
    # Perform search - correct API
    print("Performing search...")
    results = tbl.search(query_vector).limit(3).to_list()
    
    print(f"✅ Search successful! Found {len(results)} results")
    
    if results:
        first_result = results[0]
        print(f"Sample result keys: {list(first_result.keys())}")
        print(f"Question: {first_result['question'][:100]}...")
        distance_key = '_distance' if '_distance' in first_result else 'distance'
        if distance_key in first_result:
            print(f"Distance: {first_result[distance_key]:.4f}")
    
except Exception as e:
    print(f"❌ Search failed: {e}")
    import traceback
    traceback.print_exc()