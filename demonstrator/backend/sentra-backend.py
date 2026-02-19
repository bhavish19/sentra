from flask import Flask

BACKEND_VERSION="00.04.019"

import sentrabackend
from sentrabackend import log as log

class Comittee:
    m_Comittee:set[str]
    def __init__(self):
     self.m_Comittee=set()

    def addComitteeMember(self,node_id:str):
        self.m_Comittee.add(node_id)

backend:sentrabackend.Backend

if __name__ == '__main__':
    log("Starting Sentra Backend...")
    log(f"Version: {BACKEND_VERSION}")

    cmdlineargs=sentrabackend.CommandLineOptions()

    backend=sentrabackend.Backend()
    app:Flask=backend.create(cmdlineargs)
    app.run(debug=False,port=cmdlineargs.getPort(),host=cmdlineargs.getHost(),threaded=True)
