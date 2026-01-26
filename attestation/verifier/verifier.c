#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <pthread.h>

#include "sgx_quote_3.h"
#include "sgx_urts.h"
#include "sgx_pce.h"
#include "sgx_error.h"

#include "occlum_dcap.h"

#define QE_ID_SIZE 16
#define PORT 9000
#define BUF_SIZE 1024
#define MAX_CLIENTS 5

static int client_count = 0;
pthread_mutex_t client_mutex = PTHREAD_MUTEX_INITIALIZER;

/*int verify_quote(int nonce, void *quote, int quote_size){
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
}*/

void *client_handler(void *arg) {
    int client_fd = *(int *)arg;
    free(arg);

    char buffer[BUF_SIZE];
    int n = read(client_fd, buffer, BUF_SIZE - 1);

    if (n > 0) {
        buffer[n] = '\0';
        printf("Thread %lu received: %s\n", pthread_self(), buffer);
    }

    close(client_fd);

    pthread_mutex_lock(&client_mutex);
    client_count--;
    pthread_mutex_unlock(&client_mutex);

    return NULL;
}

int main() {

    uint32_t collateral_expiration_status = 1;
    sgx_ql_qv_result_t quote_verification_result = SGX_QL_QV_RESULT_UNSPECIFIED;

    int server_fd, client_fd;
    struct sockaddr_in addr;
    char buffer[BUF_SIZE];

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        return 1;
    }

    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        return 1;
    }

    listen(server_fd, 1);
    printf("Server listening on port %d\n", PORT);

    while (1) {
        pthread_mutex_lock(&client_mutex);
        if (client_count >= MAX_CLIENTS) {
            pthread_mutex_unlock(&client_mutex);
            sleep(1);
            continue;
        }
        pthread_mutex_unlock(&client_mutex);

        int *client_fd = malloc(sizeof(int));
        *client_fd = accept(server_fd, NULL, NULL);

        if (*client_fd < 0) {
            perror("accept");
            free(client_fd);
            continue;
        }

        pthread_t tid;
        pthread_create(&tid, NULL, client_handler, client_fd);
        pthread_detach(tid);

        pthread_mutex_lock(&client_mutex);
        client_count++;
        pthread_mutex_unlock(&client_mutex);
    }

    close(server_fd);
    return 0;
}