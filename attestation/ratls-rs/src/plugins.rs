extern crate occlum_dcap;
use occlum_dcap::*;
use std::convert::TryFrom;
use std::io::Result;
use std::str;

pub struct SgxDcapProvider {
    config: SgxDcapConfig,
}

pub struct SgxDcapConfig {
    /// Collateral source (PCCS URL)
    pub pccs_url: Option<String>,
    
    /// Whether to include full collateral in evidence
    pub embed_collateral: bool,
}

impl AttestationProvider for SgxDcapProvider {
    fn evidence_type(&self) -> EvidenceType {
        EvidenceType::SgxDcap
    }
    
    async fn generate_evidence(
        &self,
        nonce: &[u8],
    ) -> Result<AttestationEvidence, RAError> {
        // 1. Create report data (SHA-256 of nonce)
        let mut report_data = [0u8; 64];
        report_data[..32].copy_from_slice(nonce);
        
        // 2. Generate SGX quote
        let quote = self
        .dcap_quote
        .generate_quote(self.quote_buf.as_mut_ptr(), &mut self.req_data)
        .unwrap();
        
        // 3. Fetch collateral if needed
        let collateral = if self.config.embed_collateral {
            self.fetch_dcap_collateral(&quote).await?
        } else {
            vec![]
        };
        
        Ok(AttestationEvidence {
            data: quote,
            collateral,
        })
    }


    async fn verify_evidence(
        &self,
        evidence: &AttestationEvidence,
    ) -> Result<VerificationResult, RAError> {
        let quote = &evidence.data;
        
        // 1. Parse quote structure
        let quote_header = parse_quote_header(quote)?;
        
        // 2. Get collateral (from evidence or fetch from PCCS)
        let collateral = if evidence.collateral.is_empty() {
            self.fetch_dcap_collateral(quote).await?
        } else {
            evidence.collateral.clone()
        };
        
        let mut quote_verification_result = sgx_ql_qv_result_t::SGX_QL_QV_RESULT_UNSPECIFIED;
        let mut status = 1;

        let mut verify_arg = IoctlVerDCAPQuoteArg {
            quote_buf: self.quote_buf.as_mut_ptr(),
            quote_size: self.quote_size,
            collateral_expiration_status: &mut status,
            quote_verification_result: &mut quote_verification_result,
            supplemental_data_size: self.supplemental_size,
            supplemental_data: self.suppl_buf.as_mut_ptr(),
        };
        
        Ok(VerificationResult {
            measurements,
            tcb_status: map_dcap_status(verification_result.tcb_status),
            attestation_time: SystemTime::now(),
        })
    }
    
    fn get_measurements(&self) -> Result<Measurements, RAError> {
        // Read MRENCLAVE, MRSIGNER from current enclave
        let report = unsafe { sgx_self_report()? };
        
        Ok(Measurements {
            mr_enclave: Some(report.body.mr_enclave.m),
            mr_signer: Some(report.body.mr_signer.m),
            isv_prod_id: Some(report.body.isv_prod_id),
            isv_svn: Some(report.body.isv_svn),
            ..Default::default()
        })
    }
}

impl SgxDcapProvider {
    async fn fetch_dcap_collateral(
        &self,
        quote: &[u8],
    ) -> Result<Vec<Collateral>, RAError> {
        // Extract FMSPC, PCE ID from quote
        let fmspc = extract_fmspc(quote)?;
        let pce_id = extract_pce_id(quote)?;
        
        // Fetch from PCCS
        let pccs_url = self.config.pccs_url.as_ref()
            .ok_or(RAError::AttestationFailed(
                "No PCCS URL configured".to_string()
            ))?;
        
        let client = reqwest::Client::new();
        
        // Fetch TCB info
        let tcb_info = client
            .get(format!("{}/sgx/certification/v4/tcb", pccs_url))
            .query(&[("fmspc", &fmspc)])
            .send()
            .await?
            .bytes()
            .await?;
        
        // Fetch QE identity
        let qe_identity = client
            .get(format!("{}/sgx/certification/v4/qe/identity", pccs_url))
            .send()
            .await?
            .bytes()
            .await?;
        
        // Fetch PCK certificate chain
        let pck_certs = client
            .get(format!("{}/sgx/certification/v4/pckcert", pccs_url))
            .query(&[
                ("fmspc", &fmspc),
                ("pceid", &pce_id),
            ])
            .send()
            .await?
            .bytes()
            .await?;
        
        Ok(vec![
            Collateral::TcbInfo(tcb_info.to_vec()),
            Collateral::QeIdentity(qe_identity.to_vec()),
            Collateral::PckCertChain(pck_certs.to_vec()),
        ])
    }
}