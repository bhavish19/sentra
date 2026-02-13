//! This is the simplest possible server using rustls that does something useful:
//! it accepts the default configuration, loads a server certificate and private key,
//! and then accepts a single client connection.
//!
//! Usage: cargo r --bin simpleserver <path/to/cert.pem> <path/to/privatekey.pem>
//!
//! Note that `unwrap()` is used to deal with networking errors; this is not something
//! that is sensible outside of example code.

use ratls_rust::*;

use std::env;
use std::error::Error as StdError;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;
use std::sync::Arc;

pub async fn open_server(port: u16) -> Result<(TlsAcceptor, TcpListener), Box<dyn std::error::Error>> {
    /* todo */
    let sgx_config = SGXConfig {
        pccs_url: "10.80.1.80",
        embed_collateral: false,
    };

    let sgx = SgxDcapProvider::new(sgx_config)?;

    let dcap = RATLSServerConfigBuilder::new(dcap)
        .build()?;
    
    let acceptor = TLSAcceptor::from(Arc::new(dcap));

    let listener = TcpListener::bind(("0.0.0.0", port)).await?;

    Ok((acceptor, listener))
}

/*
1. accept client connection
*/
fn handle_connection(/* variables */) {
    
}

/*
1. Setup Rustls
2. Read pem file and cert
3. Start Server
4. Create new RATLS_RASystem
*/

fn main() -> Result<(), Box<dyn StdError>> {
    let mut args = env::args();
    args.next();
    let cert_file = args
        .next()
        .expect("missing certificate file argument");
    let private_key_file = args
        .next()
        .expect("missing private key file argument");

    /*let certs = rustls_pemfile::certs(&mut BufReader::new(&mut File::open(cert_file)?))
        .collect::<Result<Vec<_>, _>>()?;
    let private_key =
        rustls_pemfile::private_key(&mut BufReader::new(&mut File::open(private_key_file)?))?
            .unwrap();
    let config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, private_key)?;

    let listener = TcpListener::bind(format!("[::]:{}", 4443)).unwrap();
    let (mut stream, _) = listener.accept()?;

    let mut conn = rustls::ServerConnection::new(Arc::new(config))?;
    conn.complete_io(&mut stream)?;

    conn.writer()
        .write_all(b"Hello from the server")?;
    conn.complete_io(&mut stream)?;
    let mut buf = [0; 64];
    let len = conn.reader().read(&mut buf)?;
    println!("Received message from client: {:?}", &buf[..len]);
    conn.send_close_notify();
    conn.complete_io(&mut stream)?;*/


    open_server(4443);

    Ok(())
}