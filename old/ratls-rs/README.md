# RATLS-Rust (not finished!)

This is a Rust version of RA-TLS.

## Design Choices

In contrast to the C Implementation by Weinhold et al., we make use of Rustls instead of OpenSSL.
This is due to the fact that OpenSSL is C-based and therefore does not provide memory safety.

## Changes needed in Rustls

Rustls does not allow the usage of [custom extensions](https://docs.rs/rustls/latest/rustls/manual/_04_features/index.html#about-custom-extensions), as defined in TLS 1.3, as "there is no reasonable way to technically limit that API to that set of extensions. That makes the API pretty unsafe (in the TLS and cryptography sense, not memory safety sense)".
Therefore, we patch the library to allow the usage of callbacks for attestation.

## Modules

- `attest.rs`: generate attestation reports, computation of LinkingHash
- `core.rs`: 
  - `trait AttestationProvider: Send + Sync`:
    - evidence type
    - generate evidence
    - verify evidence
    - get measurements
  - `trait VerificationPolicy: Send + Sync`:
    - verify
    - describe
  - `trait LinkingHashComputer`: - does that have to be a trait?
    - computed linking hash
    - verify linking hash
- `error.rs`
- `plugins.rs`: 
  - `impl AttestationProvider for SgxDcapProvider`
- `tls.rs`: callback functions
  - 
- `verify.rs`