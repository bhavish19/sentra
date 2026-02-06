use std::ffi::c_int;
mod types;
pub use types::sgx_report_data_t;
pub use types::IoctlVerDCAPQuoteArg;

pub struct DcapQuote {
    fd: c_int,
    quote_size: u32,
    supplemental_size: u32,
}



#[derive(Debug)]
pub enum DcapError {
    NotImplemented,
}

impl DcapQuote {

    pub fn new() -> Result<Self, DcapError> {
        Err(DcapError::NotImplemented)
    }

    pub fn get_quote_size(&mut self) -> Result<u32, DcapError> {
        Err(DcapError::NotImplemented)
    }

    pub fn generate_quote(
        &mut self,
        quote_buf: *mut u8,
        report_data: *const sgx_report_data_t,
    ) -> Result<i32, DcapError> {
        Err(DcapError::NotImplemented)
    }

    pub fn get_supplemental_data_size(&mut self) -> Result<u32, DcapError> {
        Err(DcapError::NotImplemented)
    }

    pub fn verify_quote(&mut self, verify_arg: *mut IoctlVerDCAPQuoteArg) -> Result<i32, DcapError> {
        Err(DcapError::NotImplemented)
    }

    pub fn close(&mut self) {
    }

}