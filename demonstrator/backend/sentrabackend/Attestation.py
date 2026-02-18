import dcap_qvl
import time

from .Log import log as log

class Attestation:
    async def verify(self,quote:bytes)->bool:
        # Verify the quote
        log("try to verify quote...")
        now_timestamp = int(time.time())
        result:dcap_qvl.VerifiedReport
        try:
            collateral =  await dcap_qvl.get_collateral_from_pcs(quote)
            result =  dcap_qvl.verify(quote,collateral,now_timestamp)
        except Exception as e:
            log("Sone excepetion in verify")
            log(str(e))
            return False
        log(result.status)
        #ToDo - need to check the result

        parsed_quote:dcap_qvl.Quote = dcap_qvl.parse_quote(quote)
        quote_header:dcap_qvl.QuoteHeader=parsed_quote.header
        log(str(quote_header.version))
        log(str(quote_header.attestation_key_type))
        log(quote_header.user_data.hex())

        enclave_report = parsed_quote.report
        if(not isinstance(enclave_report,dcap_qvl.SgxEnclaveReport)):
            log("Not an SGX enclave")
            return False
        sgx_report:dcap_qvl.SgxEnclaveReport =enclave_report
        # Zugriff auf die wichtigsten Felder
        log(f"MRENCLAVE: {sgx_report.mr_enclave.hex()}")
        log(f"MRSIGNER:  {sgx_report.mr_signer.hex()}")
        log(f"Attributes: {sgx_report.attributes.hex()}")
        log(f"ReportData: {sgx_report.report_data.hex()}")
        return True
