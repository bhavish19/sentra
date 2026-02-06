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
    
    /// Create TLS extension containing attestation report
    pub fn create_extension(
        &self,
        report: &AttestationReport,
    ) -> Result<TlsExtension, RAError> {
        // Encode report as TLS extension following Weinhold format
        let encoded = self.encode_report(report)?;
        
        Ok(TlsExtension {
            extension_type: RATLS_EXTENSION_TYPE,
            data: encoded,
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
        
        let mut encoder = cbor::Encoder::new();
        encoder.encode_map(|map| {
            map.entry("type", report.evidence_type as u32);
            map.entry("evidence", &report.evidence);
            map.entry("linking_hash", &report.linking_hash);
            if self.config.embed_collateral {
                map.entry("collateral", &report.collateral);
            }
        })?;
        
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