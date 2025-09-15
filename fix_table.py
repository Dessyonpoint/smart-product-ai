import mysql.connector
import os
import urllib.request

# Download certificate
def download_ssl_certificate():
    cert_url = "https://letsencrypt.org/certs/isrgrootx1.pem"
    cert_path = "isrgrootx1.pem"
    if not os.path.exists(cert_path):
        urllib.request.urlretrieve(cert_url, cert_path)
    return os.path.abspath(cert_path)

cert_path = download_ssl_certificate()

# Connection config
DB_CONFIG = {
    'host': "gateway01.us-west-2.prod.aws.tidbcloud.com",
    'port': 4000,
    'user': "2jkhr6CB8XWiLJu.root",
    'password': "ReQdwdBRsZkEx2Ac",
    'database': "recommenddb",
    'ssl_ca': cert_path,
    'ssl_verify_cert': True
}

try:
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Add missing columns
    alter_sql = """
    ALTER TABLE item_embeddings 
    ADD COLUMN IF NOT EXISTS category VARCHAR(255),
    ADD COLUMN IF NOT EXISTS price DECIMAL(10,2),
    ADD COLUMN IF NOT EXISTS brand VARCHAR(255)
    """
    
    cursor.execute(alter_sql)
    conn.commit()
    print("✅ Columns added successfully!")
    
    # Verify table structure
    cursor.execute("DESCRIBE item_embeddings;")
    columns = cursor.fetchall()
    print("\n📋 Updated table structure:")
    for col in columns:
        print(f"  - {col[0]}: {col[1]}")
    
except Exception as e:
    print(f"❌ Error: {e}")
finally:
    cursor.close()
    conn.close()