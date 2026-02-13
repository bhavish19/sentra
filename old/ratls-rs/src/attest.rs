use occlum_sgx::SGXQuote;
use ciborium::ser::into_writer;
use serde::Serialize;
use std::sync::Arc;

use crate::core::*;
use crate::error::*;

#[derive(Serialize)]
struct Evidence<'a> {
    r#type: u32,
    evidence: &'a [u8],
    linking_hash: &'a [u8],
    collateral: Option<&'a [u8]>,
}

/// RATLS attester for generating attestation reports
pub struct RATLSAttester<P: AttestationProvider> {
    /// Platform-specific attestation provider
    provider: Arc<P>,
    
    /// Linking hash computer
    hash_computer: Arc<dyn LinkingHashComputer>,
    
    /// Configuration
    config: AttesterConfig,
}

pub struct AttesterConfig {
    /// Whether to include collateral in extensions
    pub embed_collateral: bool,
    
    /// Maximum extension size
    pub max_extension_size: usize,
    
    /// Evidence type to use
    pub evidence_type: EvidenceType,
    
    /// Whether to support mutual attestation
    pub mutual_attestation: bool,
}

impl<P: AttestationProvider> RATLSAttester<P> {
    /// Create new attester with provider
    pub fn new(
        provider: P,
        config: AttesterConfig,
    ) -> Self {
        Self {
            provider: Arc::new(provider),
            hash_computer: Arc::new(DefaultLinkingHashComputer),
            config,
        }
    }
    
    /// Generate attestation report for current TLS session
    /// 
    /// This is called during TLS handshake when the Certificate
    /// message is being prepared. The linking hash binds the
    /// attestation to this specific TLS session.
    pub fn generate_report(
        &self,
        session: &RASession,
    ) -> Result<AttestationReport, RAError> {
        // 1. Compute linking hash from handshake log + DHE secret
        let dhe_secret = session.dhe_secret
            .as_ref()
            .ok_or(RAError::AttestationFailed(
                "DHE secret not available".to_string()
            ))?;
        
        let linking_hash = self.hash_computer.compute_linking_hash(
            &session.handshake_log,
            dhe_secret,
        );
        
        // 2. Generate platform attestation with linking hash as nonce
        let evidence = self.provider.generate_evidence(&linking_hash)?;
        
        // 3. Construct attestation report
        Ok(AttestationReport {
            evidence_type: self.provider.evidence_type(),
            evidence: evidence.data,
            linking_hash,
            collateral: evidence.collateral,
        })
    }
    
    /// Encode report for TLS extension
    fn encode_report(
        &self,
        report: &AttestationReport,
    ) -> Result<Vec<u8>, RAError> {
        // Use CBOR encoding following interoperable RA-TLS spec
        // Structure:
        // - Evidence type tag
        // - Evidence data
        // - Linking hash
        // - Optional collateral
        
        let evidence = Evidence {
            r#type: report.evidence_type as u32,
            evidence: &report.evidence,
            linking_hash: &report.linking_hash,
            collateral: if self.config.embed_collateral {
                Some(&report.collateral)
            } else {
                None
            },
        };
        
        let mut buf = Vec::new();
        ciborium::ser::into_writer(&evidence, &mut buf)?;
        
        Ok(encoder.finish())
    }
}

/// Default linking hash implementation following Weinhold et al.
pub struct DefaultLinkingHashComputer;

impl LinkingHashComputer for DefaultLinkingHashComputer {
    fn compute_linking_hash(
        &self,
        handshake_log: &[HandshakeMessage],
        dhe_secret: &[u8],
    ) -> [u8; 32] {
        use sha2::{Sha256, Digest};
        
        let mut hasher = Sha256::new();
        
        // Hash handshake messages in order
        for msg in handshake_log {
            hasher.update(&msg.message_type.to_be_bytes());
            hasher.update(&(msg.data.len() as u32).to_be_bytes());
            hasher.update(&msg.data);
        }
        
        // Include DHE shared secret
        hasher.update(dhe_secret);
        
        hasher.finalize().into()
    }
    
    fn verify_linking_hash(
        &self,
        report: &AttestationReport,
        handshake_log: &[HandshakeMessage],
        dhe_secret: &[u8],
    ) -> Result<(), RAError> {
        let computed = self.compute_linking_hash(handshake_log, dhe_secret);
        
        if computed != report.linking_hash {
            return Err(RAError::LinkingHashMismatch);
        }
        
        Ok(())
    }
}