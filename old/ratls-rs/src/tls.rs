use rustls::{ServerConfig, ClientConfig, Certificate, ServerCertVerifier};

/// RATLS server configuration builder
pub struct RATLSServerConfigBuilder<P: AttestationProvider> {
    attester: RATLSAttester<P>,
    cert_key_pair: Option<(Vec<Certificate>, PrivateKey)>,
    mutual_auth: bool,
}

impl<P: AttestationProvider> RATLSServerConfigBuilder<P> {
    pub fn new(provider: P) -> Self {
        let config = AttesterConfig {
            embed_collateral: true,
            max_extension_size: 64 * 1024,
            evidence_type: provider.evidence_type(),
            mutual_attestation: false,
        };
        
        Self {
            attester: RATLSAttester::new(provider, config),
            cert_key_pair: None,
            mutual_auth: false,
        }
    }
    
    /// Set TLS certificate and key (for dual authentication)
    pub fn with_cert_key(
        mut self,
        cert_chain: Vec<Certificate>,
        private_key: PrivateKey,
    ) -> Self {
        self.cert_key_pair = Some((cert_chain, private_key));
        self
    }
    
    /// Enable mutual attestation (client must also attest)
    pub fn with_mutual_attestation(mut self) -> Self {
        self.mutual_auth = true;
        self
    }
    
    /// Build ServerConfig
    pub fn build(self) -> Result<ServerConfig, RAError> {
        let mut config = ServerConfig::builder()
            .with_safe_defaults()
            .with_no_client_auth()
            .with_cert_resolver(Arc::new(
                RATLSCertResolver::new(self.attester, self.cert_key_pair)
            ));
        
        // Register extension handler
        config.enable_ratls_extensions();
        
        Ok(config)
    }
}

/// Custom certificate resolver that adds RATLS extensions
struct RATLSCertResolver<P: AttestationProvider> {
    attester: RATLSAttester<P>,
    cert_key_pair: Option<(Vec<Certificate>, PrivateKey)>,
}

impl<P: AttestationProvider> ResolvesServerCert for RATLSCertResolver<P> {
    fn resolve(
        &self,
        client_hello: ClientHello,
    ) -> Option<Arc<CertifiedKey>> {
        // Get standard certificate if available
        let base_cert = self.cert_key_pair.as_ref()?;
        
        // Create certified key with RATLS extension callback
        Some(Arc::new(CertifiedKey {
            cert: base_cert.0.clone(),
            key: Arc::new(base_cert.1.clone()),
            ocsp: None,
            sct_list: None,
            
            // This callback is invoked when preparing Certificate message
            extension_callback: Some(Box::new(move |session_data| {
                // Generate attestation report
                let ra_session = extract_ra_session(session_data)?;
                let report = self.attester.generate_report(&ra_session)?;
                
                // Create extension
                self.attester.create_extension(&report)
            })),
        }))
    }
}

/// RATLS client configuration builder
pub struct RATLSClientConfigBuilder {
    verifier: RATLSVerifier,
    root_store: RootCertStore,
}

impl RATLSClientConfigBuilder {
    pub fn new(policy: impl VerificationPolicy + 'static) -> Self {
        let config = VerifierConfig {
            require_attestation: true,
            accepted_types: vec![
                EvidenceType::Tpm20,
                EvidenceType::SgxDcap,
                EvidenceType::TdxQuote,
            ],
            verify_collateral: true,
            verification_timeout: Duration::from_secs(30),
        };
        
        Self {
            verifier: RATLSVerifier::new(policy, config),
            root_store: RootCertStore::empty(),
        }
    }
    
    /// Register attestation provider
    pub fn register_provider(
        mut self,
        provider: impl AttestationProvider + 'static,
    ) -> Self {
        self.verifier.register_provider(provider);
        self
    }
    
    /// Add root CA certificate (for dual verification)
    pub fn with_root_certificates(
        mut self,
        roots: RootCertStore,
    ) -> Self {
        self.root_store = roots;
        self
    }
    
    /// Build ClientConfig
    pub fn build(self) -> Result<ClientConfig, RAError> {
        let config = ClientConfig::builder()
            .with_safe_defaults()
            .with_custom_certificate_verifier(Arc::new(
                RATLSCertVerifier::new(self.verifier, self.root_store)
            ))
            .with_no_client_auth();
        
        Ok(config)
    }
}

/// Custom certificate verifier that checks RATLS extensions
struct RATLSCertVerifier {
    verifier: RATLSVerifier,
    standard_verifier: WebPkiVerifier,
}

impl ServerCertVerifier for RATLSCertVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &Certificate,
        intermediates: &[Certificate],
        server_name: &ServerName,
        scts: &mut dyn Iterator<Item = &[u8]>,
        ocsp_response: &[u8],
        now: SystemTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        // 1. First verify standard certificate (if dual auth)
        if !self.verifier.config.require_attestation {
            self.standard_verifier.verify_server_cert(
                end_entity,
                intermediates,
                server_name,
                scts,
                ocsp_response,
                now,
            )?;
        }
        
        // 2. Extract and verify RATLS extension
        let extension = extract_ratls_extension(end_entity)
            .ok_or(rustls::Error::General(
                "No RATLS extension found".to_string()
            ))?;
        
        let report = self.verifier.parse_extension(&extension)
            .map_err(|e| rustls::Error::General(e.to_string()))?;
        
        let ra_session = get_current_ra_session()?;
        
        let result = self.verifier.verify_report(&report, &ra_session)
            .await
            .map_err(|e| rustls::Error::General(e.to_string()))?;
        
        // 3. Verification succeeded
        Ok(ServerCertVerified::assertion())
    }
}