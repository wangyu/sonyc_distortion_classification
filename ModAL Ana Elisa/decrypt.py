import pycurl
import argparse
import base64

from Crypto.Cipher import AES

try:
    from io import BytesIO
except ImportError:
    from StringIO import StringIO as BytesIO

try:
    # python 3
    from urllib.parse import urlencode
except ImportError:
    # python 2
    from urllib import urlencode


def decrypt(url,
            cacert,
            cert,
            key,
            encrypted_key_filebuf=None,
            encrypted_data_filebuf=None,
            encrypted_key_filepath=None,
            encrypted_data_filepath=None,
            output_file=None, verbose=False):
    """
    Decrypt a file or filebuffer.

    Parameters
    ----------
    url : str
        Decryption server URL
    cacert : str
        Path to CA cert
    cert : str
        Path to your cert
    key : str
        Path to key file
    encrypted_key_filebuf : BytesIO
        Buffer of encrypted data key
    encrypted_data_filebuf : BytesIO
        Buffer of encrypted data (audio)
    encrypted_key_filepath : str
        Path to encrypted data key
    encrypted_data_filepath : str
        Path to encrypted data (audio)
    output_file : str
        Path to save file to
    verbose : bool
        If True, print info

    Returns
    -------
    decrypted_data : BytesIO
    """
    # make sure that either encrypted_key buffer or file is defined
    if encrypted_key_filebuf is None and encrypted_key_filepath is None:
        raise Exception('Either `encrypted_key` or `encrypted_key_file` must be defined.')

    # make sure that enithe encrypted_data buffer or file is defined
    if encrypted_data_filebuf is None and encrypted_data_filepath is None:
        raise Exception('Either `encrypted_key` or `encrypted_key_file` must be defined.')

    buf = BytesIO()

    c = pycurl.Curl()
    c.setopt(c.POST, True)
    c.setopt(c.URL, url)
    c.setopt(c.WRITEDATA, buf)

    c.setopt(pycurl.CAINFO, cacert)

    c.setopt(pycurl.SSLCERTTYPE, "PEM")
    c.setopt(pycurl.SSLCERT, cert)

    c.setopt(pycurl.SSLKEYTYPE, "PEM")
    c.setopt(pycurl.SSLKEY, key)

    c.setopt(pycurl.SSL_VERIFYPEER, 1)
    c.setopt(pycurl.SSL_VERIFYHOST, 2)

    if encrypted_key_filepath is not None:
        with open(encrypted_key_filepath) as f:
            encrypted_key_filebuf = f.read()

    if encrypted_data_filepath is not None:
        with open(encrypted_data_filepath) as f:
            encrypted_data_filebuf = f.read()

    post_data = {
        "out_format": "raw",
        "enc": encrypted_key_filebuf
    }

    postfields = urlencode(post_data)

    c.setopt(c.POSTFIELDS, postfields)

    c.perform()

    if verbose:
        print('Status: %d' % c.getinfo(c.RESPONSE_CODE))
        print('Request Time: %f' % c.getinfo(c.TOTAL_TIME))

    decrypt_key = buf.getvalue()

    cipher = AES.new(decrypt_key)

    decrypted_data = cipher.decrypt(
        base64.b64decode(encrypted_data_filebuf)).rstrip(bytes('{', 'utf-8'))

    if output_file is not None:
        with open(output_file, 'wb') as f:
            f.write(decrypted_data)

    return BytesIO(decrypted_data)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Decrypt audo file (Example).')
    
    parser.add_argument('--url', type=str, required=True)
    parser.add_argument('--cacert', type=str, required=True)
    parser.add_argument('--cert', type=str, required=True)
    parser.add_argument('--key', type=str, required=True)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('encrypted_key', type=str)
    parser.add_argument('audio_file', type=str)
    parser.add_argument('output_file', type=str)

    args = parser.parse_args()
