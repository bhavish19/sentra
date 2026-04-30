from flask import Flask

BACKEND_VERSION="00.05.003"

import sentrabackend
from sentrabackend import log as log

backend:sentrabackend.Backend

if __name__ == '__main__':
    log("Starting Sentra Backend...")
    log(f"Version: {BACKEND_VERSION}")

    cmdlineargs=sentrabackend.CommandLineOptions()

    backend=sentrabackend.Backend()
    app:Flask=backend.create(cmdlineargs)
    app.run(debug=False,port=cmdlineargs.getPort(),host=cmdlineargs.getHost(),threaded=True)
