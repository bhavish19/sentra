use occlum_sgx::SGXQuote;

 fn main() -> Result<(), Box<dyn std::error::Error>> {

    // 1. Define your 64-byte report data
   let report_data = [0u8; 64];
    // 2. Generate the SGX Quote

    let quote = SGXQuote::from_report_data(&report_data)?;
    // 3. (Optional) Inspect the quote metadata

    println!("MrEnclave: {:?}", quote.mrenclave());
    println!("MrSigner:  {:?}", quote.mrsigner());
    // 4. Convert to bytes for transmission to a remote verifier

    let _quote_raw = quote.as_slice();
    Ok(())

}

