# Attestation using Occlum

The C program for attestation using dcap can be found under `dcap/`. It is based on [this](https://github.com/occlum/occlum/tree/master/demos/remote_attestation/dcap) example provided by Occlum.

The program currently retrieves and prints the following fields:
- **SGX ISV Family ID** (not important)
- **SGX ISV EXT Product ID** (not important)
- **SGX CONFIG ID** (not important)
- **SGX CONFIG SVN**: (not important)
- **MRENCLAVE**: measurement of the target.
- **MRSIGNER**: The Enclave Author’s Public Key – After an enclave is successfully initialized, the CPU records a hash of the enclave author’s public key in the MRSIGNER register. 
- **Extracted QE_ID**: unique identifier inserted by the Quoting Enclave derived from the platform's unique hardware sealing key. Note: if the platform owner wipes the SGX provisioning data or re-installs the Quoting Enclave, this ID might change. In the final version, it might make sense to check the 

## Running (using occlum container)

Launch occlum container by using:

`docker run -it --device /dev/sgx/enclave --device /dev/sgx/provision --device /dev/sgx_provision occlum/occlum:latest-ubuntu20.04 bash`

Configuration of `/etc/sgx_default_qcnl.conf`:
- Configure PCCS URL (currently running on `10.80.1.3`): `"pccs_url": "https://10.80.1.3:8081/sgx/certification/v4/"`
- set use_secure_cert to false: `"use_secure_cert": false`

Basic DCAP test inside the container:
```bash
cd /root/demos/remote_attestation/dcap
./run_dcap_quote_on_occlum.sh
```

## SGX specific attestation details

The **Quoting Enclave** verifies the reports that have been created to its MRENCVLAVE measurement value and then convers and signs them using a device specific asymmetric key, the Intel EPID key.

## Documentation
[Intel DCAP](https://download.01.org/intel-sgx/latest/dcap-latest/linux/docs/SGX_DCAP_Caching_Service_Design_Guide.pdf)


[Intel SGX](https://download.01.org/intel-sgx/latest/linux-latest/docs/Intel_SGX_Developer_Guide.pdf)
