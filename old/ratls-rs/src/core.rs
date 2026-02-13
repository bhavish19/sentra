/// Remote attestation session state
pub struct RASession {
    /// Unique session identifier
    pub session_id: SessionId,
    
    /// Current protocol state
    pub state: RAState,
    
    /// Handshake message log (for linking hash)
    pub handshake_log: Vec<HandshakeMessage>,
    
    /// DHE shared secret (when available)
    pub dhe_secret: Option<Vec<u8>>,
    
    /// Received attestation report
    pub peer_report: Option<AttestationReport>,
    
    /// Local attestation report (if mutual auth)
    pub local_report: Option<AttestationReport>,
    
    /// Session resumption ticket binding
    pub ticket_binding: Option<TicketBinding>,
}

/// Attestation report structure
pub struct AttestationReport {
    /// Type of evidence (TPM, SGX, etc.)
    pub evidence_type: EvidenceType,
    
    /// Raw attestation evidence
    pub evidence: Vec<u8>,
    
    /// Linking hash that binds to TLS session
    pub linking_hash: [u8; 32],
    
    /// Additional collateral (certificates, TCB info, etc.)
    pub collateral: Vec<Collateral>,
}

/// Evidence types (following Weinhold plugin model)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceType {
    /// TPM 2.0 Quote
    Tpm20,
    
    /// Intel SGX Quote (DCAP)
    SgxDcap,
    
    /// Intel TDX Quote
    TdxQuote,
    
    /// AMD SEV-SNP Report
    SevSnp,
    
    /// Arm Confidential Compute Architecture
    ArmCca,
    
    /// Custom/extensible
    Custom(u32),
}

/// State machine states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RAState {
    /// Initial state
    Init,
    
    /// Extension sent/received in ClientHello
    ClientHelloSent,
    
    /// Extension sent/received in Certificate
    CertificateSent,
    
    /// Verification in progress
    Verifying,
    
    /// Verification completed successfully
    Verified,
    
    /// Session resumed from ticket
    Resumed,
    
    /// Error state
    Failed(RAError),
}

/// Platform attestation provider (TEE plugin interface)
pub trait AttestationProvider: Send + Sync {
    /// Get the evidence type this provider supports
    fn evidence_type(&self) -> EvidenceType;
    
    /// Generate attestation evidence with given nonce/challenge
    fn generate_evidence(
        &self,
        nonce: &[u8],
    ) -> Result<AttestationEvidence, RAError>;
    
    /// Verify attestation evidence
    fn verify_evidence(
        &self,
        evidence: &AttestationEvidence,
    ) -> Result<VerificationResult, RAError>;
    
    /// Get platform measurements/identity
    fn get_measurements(&self) -> Result<Measurements, RAError>;
}

/// Attestation verification policy
pub trait VerificationPolicy: Send + Sync {
    /// Verify measurements against policy
    fn verify(
        &self,
        measurements: &Measurements,
        evidence_type: EvidenceType,
    ) -> Result<(), PolicyViolation>;
    
    /// Get policy description for logging
    fn describe(&self) -> String;
}

/// Linking hash computation (critical security component)
pub trait LinkingHashComputer {
    /// Compute linking hash from handshake log and DHE secret
    /// This is the core security property that prevents relay attacks
    fn compute_linking_hash(
        &self,
        handshake_log: &[HandshakeMessage],
        dhe_secret: &[u8],
    ) -> [u8; 32];
    
    /// Verify linking hash matches expected value
    fn verify_linking_hash(
        &self,
        report: &AttestationReport,
        handshake_log: &[HandshakeMessage],
        dhe_secret: &[u8],
    ) -> Result<(), RAError>;
}