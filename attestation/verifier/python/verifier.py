import asyncio
import time
import dcap_qvl
import requests


response = requests.get("http://10.80.1.54:3000/attest")
quote_bytes =  response.content

# Verify the quote
now_timestamp = int(time.time())
collateral =  asyncio.run( dcap_qvl.get_collateral_from_pcs(quote_bytes))
result =  dcap_qvl.verify(quote_bytes,collateral,now_timestamp)

print(result)