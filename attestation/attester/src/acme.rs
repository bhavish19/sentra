use instant_acme::{
    Account, ChallengeType, Identifier, NewAccount, NewOrder, OrderStatus,
};
use std::time::Duration;
use std::fs;
use hyper_rustls::HttpsConnectorBuilder;
use hyper_util::client::legacy::Client;
use rustls::{ClientConfig, RootCertStore};

/*use acme_lib::{Account, Directory, DirectoryUrl};
use acme_lib::persist::MemoryPersist;
use acme_lib::create_p384_key;
use std::env;
*///use rustls_pki_types::CertificateDer;
 
pub fn get_tls_certificate() -> Result<(String, String), Box<dyn std::error::Error>> {
    let result= tokio::runtime::Runtime::new()?.block_on(async {
        get_tlscertificate_async().await
    });
    println!("block_on returned: {:?}", result.is_ok());
    result
}
pub async fn get_tlscertificate_async()-> Result<(String, String), Box<dyn std::error::Error>>
{
// Load root certificate
    let pem = fs::read("pebble.cer")?;
    let certs = rustls_pemfile::certs(&mut &pem[..])
        .collect::<Result<Vec<_>, _>>()?;
    println!("Certificate bytes: {:?}", certs);
    
    let mut root_store = RootCertStore::empty();
    for cert in certs {
        root_store.add(cert)?;
    }
    
    // Build rustls config
    let tls_config = ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();
    
    // Create HTTPS connector
    let https = HttpsConnectorBuilder::new()
        .with_tls_config(tls_config)
        .https_only()
        .enable_http1()
        .build();
    
    // Create hyper client
    let http_client = Client::builder(hyper_util::rt::TokioExecutor::new())
        .build(https);

    // Create account
    let (account, _) = Account::create_with_http(
        &NewAccount {
            contact: &[],
            terms_of_service_agreed: true,
            only_return_existing: false,
        },
        "https://127.0.0.1:14000/dir",
        None,
        Box::new(http_client)
    )
    .await?;

    let params = rcgen::CertificateParams::new(vec!["example.com".to_string()])?;
    let key_pair = rcgen::KeyPair::generate()?;
    
    let private_key = key_pair.serialize_pem();
    let csr = params.serialize_request(&key_pair)?;
    let csr_der = csr.der().to_vec();
    // Create order
    let mut order = account
        .new_order(&NewOrder {
            identifiers: &[Identifier::Dns("example.com".to_string())],
        })
        .await?;

    println!("Certificate Order placed...");
println!("Order created, state: {:?}", order.state());
println!("Authorizations URL: {:?}", order.state().authorizations);

println!("Fetching authorizations...");
let authorizations = order.authorizations().await?;
println!("Got {} authorizations", authorizations.len());
    println!("Recevied authorisations...");
     for authz in &authorizations {
        let challenge = authz
        .challenges
        .iter()
        .find(|c| matches!(c.r#type, ChallengeType::Http01 ))
        .ok_or("No supported challenge type found")?;
    
    if challenge.r#type == ChallengeType::Http01 {
        order.set_challenge_ready(&challenge.url).await?;
    }
    }

    tokio::time::sleep(Duration::from_millis(1000)).await;
    while order.state().status == OrderStatus::Pending {
        tokio::time::sleep(Duration::from_secs(1)).await;
        order.refresh().await?;
    }
    
    // Check if ready
    if order.state().status != OrderStatus::Ready {
        return Err(format!("Order not ready: {:?}", order.state()).into());
    }
    // Finalize with CSR (this waits for validation internally)
    println!("Try to finalise order...");
    order.finalize(&csr_der).await?;
    println!("Order finalised...");

    // Wait for certificate to be ready
    while order.state().status == OrderStatus::Processing {
        tokio::time::sleep(Duration::from_secs(1)).await;
        order.refresh().await?;
    }

    // Get certificate
    match order.certificate().await? {
    Some(cert_chain) => Ok((cert_chain, private_key)),
    None => Err("Certificate not ready".into())
}
}


/* 
pub fn get_tls_certificate(
 ) -> Result<(String, String), Box<dyn std::error::Error>> {
    env::set_var("SSL_CERT_FILE", "pebble.cer");
    let persist = MemoryPersist::new();
    let url = DirectoryUrl::Other("https://127.0.0.1:14000");
    let dir = Directory::from_url(persist, url)?;
    
    let acc = dir.account("admin@example.com")?;
    
    
   // Order a new TLS certificate for a domain.
let mut ord_new = acc.new_order("mydomain.io", &[])?;

// If the ownership of the domain(s) have already been
// authorized in a previous order, you might be able to
// skip validation. The ACME API provider decides.
let ord_csr = loop {
    // are we done?
    if let Some(ord_csr) = ord_new.confirm_validations() {
        break ord_csr;
    }

    // Get the possible authorizations (for a single domain
    // this will only be one element).
    let auths = ord_new.authorizations()?;

    // For HTTP, the challenge is a text file that needs to
    // be placed in your web server's root:
    //
    // /var/www/.well-known/acme-challenge/<token>
    //
    // The important thing is that it's accessible over the
    // web for the domain(s) you are trying to get a
    // certificate for:
    //
    // http://mydomain.io/.well-known/acme-challenge/<token>
    let chall = auths[0].http_challenge();

    // The token is the filename.
    let token = chall.http_token();
    let path = format!(".well-known/acme-challenge/{}", token);

    // The proof is the contents of the file
    let proof = chall.http_proof();

    // Here you must do "something" to place
    // the file/contents in the correct place.
    // update_my_web_server(&path, &proof);

    // After the file is accessible from the web,
    // this tells the ACME API to start checking the
    // existence of the proof.
    //
    // The order at ACME will change status to either
    // confirm ownership of the domain, or fail due to the
    // not finding the proof. To see the change, we poll
    // the API with 5000 milliseconds wait between.
    chall.validate(5000)?;

    // Update the state against the ACME API.
    ord_new.refresh()?;
};

// Ownership is proven. Create a private key for
// the certificate. These are provided for convenience, you
// can provide your own keypair instead if you want.
let pkey_pri = create_p384_key();

// Submit the CSR. This causes the ACME provider to enter a
// state of "processing" that must be polled until the
// certificate is either issued or rejected. Again we poll
// for the status change.
let ord_cert =
    ord_csr.finalize_pkey(pkey_pri, 5000)?;

// Now download the certificate. Also stores the cert in
// the persistence.
let cert = ord_cert.download_and_save_cert()?;
    
    Ok((
        cert.certificate().to_string(),
        cert.private_key().to_string()
    ))
}
    */