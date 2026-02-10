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
print(result.status)

parsed_quote = dcap_qvl.parse_quote(quote_bytes) 
print(parsed_quote.header.version)
print(parsed_quote.header.attestation_key_type)
print(parsed_quote.header.user_data)

enclave_report = parsed_quote.report

# Zugriff auf die wichtigsten Felder
print(f"MRENCLAVE: {enclave_report.mr_enclave.hex()}")
print(f"MRSIGNER:  {enclave_report.mr_signer.hex()}")
print(f"Attributes: {enclave_report.attributes.hex()}")
print(f"ReportData: {enclave_report.report_data.hex()}")


