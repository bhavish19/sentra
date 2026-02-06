// dcap-api/src/types.rs
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct sgx_report_data_t {
    pub d: [u8; 64],
}

// Copy from occlum/src/libos/src/fs/dev_fs/dev_sgx/mod.rs
//#[allow(dead_code)]
#[repr(C)]
pub struct IoctlVerDCAPQuoteArg {
    pub quote_buf: *const u8,                               // Input
    pub quote_size: u32,                                    // Input
    pub collateral_expiration_status: *mut u32,             // Output
    pub quote_verification_result: *mut sgx_ql_qv_result_t, // Output
    pub supplemental_data_size: u32,                        // Input (optional)
    pub supplemental_data: *mut u8,                         // Output (optional)
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum sgx_ql_qv_result_t {
    SGX_QL_QV_RESULT_OK = 0,
    SGX_QL_QV_RESULT_CONFIG_NEEDED = 1,
    SGX_QL_QV_RESULT_OUT_OF_DATE = 2,
    SGX_QL_QV_RESULT_OUT_OF_DATE_CONFIG_NEEDED = 3,
    SGX_QL_QV_RESULT_INVALID_SIGNATURE = 4,
    SGX_QL_QV_RESULT_REVOKED = 5,
    SGX_QL_QV_RESULT_UNSPECIFIED = 0xFFFF_FFFF,
}