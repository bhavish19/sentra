#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgx_quote_3.h"
#include "sgx_urts.h"
#include "sgx_pce.h"
#include "sgx_error.h"

#include "occlum_dcap.h"

#define QE_ID_SIZE 16

void dump_quote_info(sgx_quote3_t *p_quote)
{
    unsigned int i;
    sgx_report_body_t *p_rep_body;
    sgx_report_data_t *p_rep_data;
    sgx_ql_auth_data_t *p_auth_data;
    sgx_ql_ecdsa_sig_data_t *p_sig_data;
    sgx_ql_certification_data_t *p_cert_data;
    uint64_t*pll;

    p_rep_body = (sgx_report_body_t *)(&p_quote->report_body);
    p_rep_data = (sgx_report_data_t *)(&p_rep_body->report_data);
    p_sig_data = (sgx_ql_ecdsa_sig_data_t *)p_quote->signature_data;
    p_auth_data = (sgx_ql_auth_data_t*)p_sig_data->auth_certification_data;
    p_cert_data = (sgx_ql_certification_data_t *)((uint8_t *)p_auth_data + sizeof(*p_auth_data) + p_auth_data->size);

    printf("cert_key_type = 0x%x\n", p_cert_data->cert_key_type);

    printf("\nSGX ISV Family ID:\n");
    pll = (uint64_t *)p_rep_body->isv_family_id;
    printf("\tLow 8 bytes: \t0x%08lx\n", *pll++);
    printf("\tHigh 8 bytes: \t0x%08lx\n", *pll);

    printf("\nSGX ISV EXT Product ID:\n");
    pll = (uint64_t *)p_rep_body->isv_ext_prod_id;
    printf("\tLow 8 bytes: \t0x%08lx\n", *pll++);
    printf("\tHigh 8 bytes: \t0x%08lx\n", *pll);

    printf("\nSGX CONFIG ID:");
    for (i = 0; i < SGX_CONFIGID_SIZE; i++) {
        if (!(i % 16))
            printf("\n\t");
        printf("%02x ", p_rep_body->config_id[i]);
    }

    printf("\n\nSGX CONFIG SVN:\n");
    printf("\t0x%04x\n", p_rep_body->config_svn);

    const sgx_measurement_t *mrenclave = &p_rep_body->mr_enclave;
    const sgx_measurement_t *mrsigner = &p_rep_body->mr_signer;

    // 4. Print the value (MRENCLAVE is always 32 bytes/SHA-256)
    printf("MRENCLAVE: ");
    for (int i = 0; i < SGX_HASH_SIZE; i++) {
        printf("%02x", mrenclave->m[i]);
    }
    printf("\n");

        printf("MRSIGNER: ");
    for (int i = 0; i < SGX_HASH_SIZE; i++) {
        printf("%02x", mrsigner->m[i]);
    }
    printf("\n");

    const uint8_t *qe_id = p_quote->header.user_data;

    // Print the QE_ID
    printf("Extracted QE_ID (16 bytes): ");
    for (int i = 0; i < QE_ID_SIZE; i++) {
        printf("%02x", qe_id[i]);
    }
    printf("\n");
}

void main() {
    void *handle;
    uint32_t quote_size, supplemental_size;
    uint8_t *p_quote_buffer, *p_supplemental_buffer;
    sgx_quote3_t *p_quote;
    sgx_report_body_t *p_rep_body;
    sgx_report_data_t *p_rep_data;
    int32_t ret;
    
    handle = dcap_quote_open();
    quote_size = dcap_get_quote_size(handle);
    printf("quote size = %d\n", quote_size);

    p_quote_buffer = (uint8_t*)malloc(quote_size);
    if (NULL == p_quote_buffer) {
        printf("Couldn't allocate quote_buffer\n");
        goto CLEANUP;
    }
    memset(p_quote_buffer, 0, quote_size);

    sgx_report_data_t report_data = { 0 };
    char *data = "ioctl DCAP report data example";
    memcpy(report_data.d, data, strlen(data));

    // Get the Quote
    ret = dcap_generate_quote(handle, p_quote_buffer, &report_data);
    if (0 != ret) {
        printf( "Error in dcap_generate_quote.\n");
        goto CLEANUP;
    }

    printf("DCAP generate quote successfully\n");

    p_quote = (sgx_quote3_t *)p_quote_buffer;
    p_rep_body = (sgx_report_body_t *)(&p_quote->report_body);
    p_rep_data = (sgx_report_data_t *)(&p_rep_body->report_data);

    if (memcmp((void *)p_rep_data, (void *)&report_data, sizeof(sgx_report_data_t)) != 0) {
        printf("mismathced report data\n");
        goto CLEANUP;
    }

    printf("STORE\n");

    FILE *f = fopen("quote.dat", "wb");
    if (!f) {
        perror("fopen");
        exit(1);
    }
    fwrite(p_quote_buffer, 1, quote_size, f);
    fclose(f);
    printf("Saved quote to quote.dat (%d bytes)\n", quote_size);

    dump_quote_info(p_quote);

    supplemental_size = dcap_get_supplemental_data_size(handle);
    printf("supplemental_size size = %d\n", supplemental_size);
    p_supplemental_buffer = (uint8_t *)malloc(supplemental_size);
    if (NULL == p_supplemental_buffer) {
        printf("Couldn't allocate supplemental buffer\n");
        goto CLEANUP;
    }
    memset(p_supplemental_buffer, 0, supplemental_size);

    uint32_t collateral_expiration_status = 1;
    sgx_ql_qv_result_t quote_verification_result = SGX_QL_QV_RESULT_UNSPECIFIED;

    ret = dcap_verify_quote(
        handle,
        p_quote_buffer,
        quote_size,
        &collateral_expiration_status,
        &quote_verification_result,
        supplemental_size,
        p_supplemental_buffer
        );
    
    if (0 != ret) {
        printf( "Error in dcap_verify_quote.\n");
        goto CLEANUP;
    }

    if (collateral_expiration_status != 0) {
        printf("the verification collateral has expired\n");
    }

    switch (quote_verification_result) {
        case SGX_QL_QV_RESULT_OK:
            printf("Succeed to verify the quote!\n");
            break;
        case SGX_QL_QV_RESULT_CONFIG_NEEDED:
        case SGX_QL_QV_RESULT_OUT_OF_DATE:
        case SGX_QL_QV_RESULT_OUT_OF_DATE_CONFIG_NEEDED:
        case SGX_QL_QV_RESULT_SW_HARDENING_NEEDED:
        case SGX_QL_QV_RESULT_CONFIG_AND_SW_HARDENING_NEEDED:
            printf("WARN: App: Verification completed with Non-terminal result: %x\n",
                   quote_verification_result);
            break;
        case SGX_QL_QV_RESULT_INVALID_SIGNATURE:
        case SGX_QL_QV_RESULT_REVOKED:
        case SGX_QL_QV_RESULT_UNSPECIFIED:
        default:
            printf("\tError: App: Verification completed with Terminal result: %x\n",
                   quote_verification_result);
            goto CLEANUP;
    }

    printf("DCAP verify quote successfully\n");

CLEANUP:
    if (NULL != p_quote_buffer) {
        free(p_quote_buffer);
    }

    if (NULL != p_supplemental_buffer) {
        free(p_supplemental_buffer);
    }

    dcap_quote_close(handle);
}
