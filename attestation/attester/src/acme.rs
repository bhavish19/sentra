pub fn getTLSCertificate()
{
    // Create account
    let (account, _) = Account::create(
        &NewAccount {
            contact: &[],
            terms_of_service_agreed: true,
            only_return_existing: false,
        },
        "https://localhost:14000/dir",
        None,
    )
    .await?;

    // Generate private key and CSR first
    let mut params = rcgen::CertificateParams::new(vec!["example.com".to_string()]);
    let cert = rcgen::Certificate::from_params(params)?;
    let private_key = cert.serialize_private_key_pem();
    let csr = cert.serialize_request_der()?;

    // Create order
    let mut order = account
        .new_order(&NewOrder {
            identifiers: &[Identifier::Dns("example.com".to_string())],
        })
        .await?;

    // Answer challenges
    let authorizations = order.authorizations().await?;
    for authz in &authorizations {
        let challenge = authz.challenges.iter()
            .find(|c| c.r#type == ChallengeType::Http01)
            .ok_or("no HTTP-01")?;
        order.set_challenge_ready(&challenge.url).await?;
    }

    // Finalize with CSR (this waits for validation internally)
    order.finalize(&csr).await?;

    // Wait for certificate to be ready
    while order.state().status == OrderStatus::Processing {
        tokio::time::sleep(Duration::from_secs(1)).await;
        order.refresh().await?;
    }

    // Get certificate
    let cert_chain = order.certificate().await?;
}