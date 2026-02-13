use crate::core::*;

#[derive(Debug, thiserror::Error)]

pub enum RAError {
    #[error("Platform attestation failed: {0}")]
    AttestationFailed(String),
    
    #[error("Evidence verification failed: {0}")]
    VerificationFailed(String),
    
    #[error("Policy violation: {0}")]
    PolicyViolation(#[from] PolicyViolation),
    
    #[error("Linking hash mismatch - possible relay attack")]
    LinkingHashMismatch,
    
    #[error("Invalid state transition: {from:?} -> {to:?}")]
    InvalidStateTransition { from: RAState, to: RAState },
    
    #[error("Extension parse error: {0}")]
    ExtensionParseError(String),
    
    #[error("TLS integration error: {0}")]
    TlsError(String),
}

#[derive(Debug, thiserror::Error)]
pub enum PolicyViolation {
    #[error("TCB level too low: {actual} < {required}")]
    TcbTooLow { actual: u32, required: u32 },
    
    #[error("Measurement mismatch: {field}")]
    MeasurementMismatch { field: String },
    
    #[error("Platform not trusted: {reason}")]
    PlatformNotTrusted { reason: String },
}