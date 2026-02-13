/// RATLS verifier for checking attestation reports
pub struct RATLSVerifier {
    /// Map of evidence types to providers
    providers: HashMap<EvidenceType, Arc<dyn AttestationProvider>>,
    
    /// Verification policy
    policy: Arc<dyn VerificationPolicy>,
    
    /// Linking hash computer
    hash_computer: Arc<dyn LinkingHashComputer>,
    
    /// Configuration
    config: VerifierConfig,
}

pub struct VerifierConfig {
    /// Whether to require attestation (vs optional)
    pub require_attestation: bool,
    
    /// Accepted evidence types
    pub accepted_types: Vec<EvidenceType>,
    
    /// Whether to verify collateral
    pub verify_collateral: bool,
    
    /// Timeout for verification operations
    pub verification_timeout: Duration,
}

impl RATLSVerifier {
    /// Create new verifier
    pub fn new(
        policy: impl VerificationPolicy + 'static,
        config: VerifierConfig,
    ) -> Self {
        Self {
            providers: HashMap::new(),
            policy: Arc::new(policy),
            hash_computer: Arc::new(DefaultLinkingHashComputer),
            config,
        }
    }
    
    /// Register attestation provider for evidence type
    pub fn register_provider(
        &mut self,
        provider: impl AttestationProvider + 'static,
    ) {
        let evidence_type = provider.evidence_type();
        self.providers.insert(evidence_type, Arc::new(provider));
    }
    
    /// Verify attestation report from peer
    pub async fn verify_report(
        &self,
        report: &AttestationReport,
        session: &RASession,
    ) -> Result<VerificationResult, RAError> {
        // 1. Verify linking hash
        let dhe_secret = session.dhe_secret
            .as_ref()
            .ok_or(RAError::VerificationFailed(
                "DHE secret not available".to_string()
            ))?;
        
        self.hash_computer.verify_linking_hash(
            report,
            &session.handshake_log,
            dhe_secret,
        )?;
        
        // 2. Get provider for evidence type
        let provider = self.providers
            .get(&report.evidence_type)
            .ok_or(RAError::VerificationFailed(
                format!("No provider for {:?}", report.evidence_type)
            ))?;
        
        // 3. Verify evidence
        let evidence = AttestationEvidence {
            data: report.evidence.clone(),
            collateral: report.collateral.clone(),
        };
        
        let result = provider.verify_evidence(&evidence).await?;
        
        // 4. Apply policy
        self.policy.verify(&result.measurements, report.evidence_type)?;
        
        Ok(result)
    }
    
    /// Parse extension into attestation report
    pub fn parse_extension(
        &self,
        extension: &TlsExtension,
    ) -> Result<AttestationReport, RAError> {
        if extension.extension_type != RATLS_EXTENSION_TYPE {
            return Err(RAError::ExtensionParseError(
                "Wrong extension type".to_string()
            ));
        }
        
        self.decode_report(&extension.data)
    }
    
    /// Decode CBOR-encoded report
    fn decode_report(&self, data: &[u8]) -> Result<AttestationReport, RAError> {
        let decoder = cbor::Decoder::new(data);
        let map = decoder.decode_map()?;
        
        let evidence_type = EvidenceType::from_u32(
            map.get("type")
                .ok_or(RAError::ExtensionParseError("Missing type".to_string()))?
        );
        
        let evidence = map.get("evidence")
            .ok_or(RAError::ExtensionParseError("Missing evidence".to_string()))?
            .to_vec();
        
        let linking_hash: [u8; 32] = map.get("linking_hash")
            .ok_or(RAError::ExtensionParseError("Missing linking_hash".to_string()))?
            .try_into()
            .map_err(|_| RAError::ExtensionParseError("Invalid hash length".to_string()))?;
        
        let collateral = map.get("collateral")
            .map(|c| c.to_vec())
            .unwrap_or_default();
        
        Ok(AttestationReport {
            evidence_type,
            evidence,
            linking_hash,
            collateral,
        })
    }
}