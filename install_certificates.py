#!/usr/bin/env python3
# Script to install certificates for Python on macOS
# Based on https://stackoverflow.com/questions/50236117/

import os
import ssl
import stat
import subprocess
from urllib.request import urlopen

# Download and install certifi
def install_certifi():
    print("Installing certifi...")
    try:
        import pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "certifi"])
        print("certifi installed successfully")
    except Exception as e:
        print(f"Error installing certifi: {e}")
        print("Please run: pip install --upgrade certifi")

# Download and install certificates
def install_cert():
    print("Installing certificates...")
    # For built-in Python on macOS
    STAT_0o775 = (stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
                | stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP
                | stat.S_IROTH | stat.S_IXOTH)
    
    openssl_dir, openssl_cafile = os.path.split(
        ssl.get_default_verify_paths().openssl_cafile)
    
    print(f" - Certificates directory: {openssl_dir}")
    print(f" - Certificate file: {openssl_cafile}")
    
    if not os.path.exists(openssl_dir):
        os.makedirs(openssl_dir)
    
    # Download current CA bundle from certifi
    import certifi
    certifi_cafile = certifi.where()
    print(f" - Using certifi from: {certifi_cafile}")
    
    # Copy certifi CA bundle to the OpenSSL directory
    with open(certifi_cafile, 'rb') as infile:
        with open(os.path.join(openssl_dir, openssl_cafile), 'wb') as outfile:
            outfile.write(infile.read())
    
    os.chmod(os.path.join(openssl_dir, openssl_cafile), STAT_0o775)
    print(" - Certificate installation complete!")

if __name__ == '__main__':
    import sys
    
    # Install certifi package first
    install_certifi()
    
    # Then install certificates
    try:
        install_cert()
        
        # Test if it works
        print("\nTesting SSL connection...")
        response = urlopen('https://www.python.org')
        print("SSL connection successful!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nAlternative solution:")
        print("Add the following code before downloading models:")
        print("import ssl")
        print("ssl._create_default_https_context = ssl._create_unverified_context") 