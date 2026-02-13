# Examples

Current state: simple Rust TLS Server - Client

## Setup

Generate a self-signed certificate for the server and additionally convert it to `.der` format:
```bash
$ openssl req -x509 -newkey rsa:4096 -sha256 -days 365 -nodes \
  -keyout key.pem -out cert.pem \
  -subj "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost" \
  -addext "basicConstraints=critical,CA:FALSE" \
  -addext "keyUsage=critical,digitalSignature,keyEncipherment" \
  -addext "extendedKeyUsage=serverAuth"
$ openssl x509 -in cert.pem -outform der -out server_cert.der
```

## Building

```bash
$ cargo build --example
```

## Running

1. Server

```bash
$ cargo run --example sgx-server cert.pem key.pem
```
1. Client
   
```bash
$ cargo run --example sgx-client
```  