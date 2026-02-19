#For getting TLS certificate using ACME
import josepy as jose
import acme.client
import acme.messages
from acme import challenges
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.hazmat.primitives import serialization
from acme import crypto_util
import socket

from .Log import log as log

class SentraACME:

    m_acmeHost:str
    m_acmeCert:str|None
    def __init__(self,acmeHost:str,acmeCert:str|None):
        self.m_acmeCert=acmeCert
        self.m_acmeHost=acmeHost

    def generateTLSCertsAndKeys(self)->tuple[bytes|None,bytes|None]:
        try:
            log("Try to get TLS certificate...")
            acc_key = jose.JWKRSA(key=rsa.generate_private_key(65537, 2048, default_backend()))
            # Connect and register (single account creation)
            acmeCert:str|bool
            if(self.m_acmeCert is None):
                acmeCert=False
            else:
                acmeCert=self.m_acmeCert
            net = acme.client.ClientNetwork(acc_key, verify_ssl=acmeCert)
            strURL:str=f"https://{self.m_acmeHost}:14000/dir"
            directory = acme.client.ClientV2.get_directory(strURL, net)
            _acme:acme.client.ClientV2 = acme.client.ClientV2(directory, net=net)
            _acme.new_account(acme.messages.NewRegistration.from_data(terms_of_service_agreed=True))

            # Generate private key for certificate
            cert_key:RSAPrivateKey = rsa.generate_private_key(65537, 2048, default_backend())
            key_pem:bytes = cert_key.private_bytes(serialization.Encoding.PEM,
                                            serialization.PrivateFormat.TraditionalOpenSSL,
                                                serialization.NoEncryption()
                                            )
            hostname:str=socket.gethostname()
            csr_pem:bytes = crypto_util.make_csr(key_pem, [hostname])
            # Order certificate
            order:acme.messages.OrderResource = _acme.new_order(csr_pem)
            for authz in order.authorizations:
                # Try to find a supported challenge type
                challenge = None

                for chall in authz.body.challenges:
                    if isinstance(chall.chall, challenges.HTTP01):
                        challenge = chall
                        break
                    elif isinstance(chall.chall, challenges.DNS01):
                        challenge = chall
                        break

                if challenge:
                    response = challenge.response(acc_key)
                    _acme.answer_challenge(challenge, response)

            order = _acme.poll_and_finalize(order)
            log("Got TLS certificate.")
            return (key_pem,order.fullchain_pem.encode('utf-8'))
        except:
            log("Failure in getting TLS certificate - continue without TLS.")
            return (None,None)
