#code heaviliy borrowed from https://github.com/anibali/h36m-fetch

from subprocess import call
from os import path, makedirs
import hashlib
from tqdm import tqdm
import configparser
import requests
import tarfile
from glob import glob


BASE_URL = 'http://vision.imar.ro/human3.6m/filebrowser.php'

subjects = [
    ('S1', 1),
    ('S5', 6),
    ('S6', 7),
    ('S7', 2),
    ('S8', 3),
    ('S9', 4),
    ('S11', 5),
]


def md5(filename):
    hash_md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_file(url, dest_file, phpsessid):
    call(['axel',
          '-a',
          '-n', '24',
          '-H', 'COOKIE: PHPSESSID=' + phpsessid,
          '-o', dest_file,
          url])

def get_config():
    dirpath = path.dirname(path.realpath(__file__))
    config = configparser.ConfigParser()
    config.read(path.join(dirpath,'config.ini'))
    return config


def get_phpsessid(config):

    try:
        phpsessid = config['General']['PHPSESSID']
    except (KeyError, configparser.NoSectionError):
        print('Could not read PHPSESSID from `config.ini`.')
        phpsessid = input('Enter PHPSESSID: ')
    return phpsessid


def verify_phpsessid(phpsessid):
    requests.packages.urllib3.disable_warnings()
    test_url = 'http://vision.imar.ro/human3.6m/filebrowser.php'
    resp = requests.get(test_url, verify=False, cookies=dict(PHPSESSID=phpsessid))
    fail_message = 'Failed to verify your PHPSESSID. Please ensure that you ' \
                   'are currently logged in at http://vision.imar.ro/human3.6m/ ' \
                   'and that you have copied the PHPSESSID cookie correctly.'
    assert resp.url == test_url, fail_message


def download_all(phpsessid, out_dir):
    checksums = {}
    dirpath = path.dirname(path.realpath(__file__))
    with open(path.join(dirpath,'checksums.txt'), 'r') as f:
        for line in f.read().splitlines(keepends=False):
            v, k = line.split('  ')
            checksums[k] = v

    files = []
    for subject_id, id in subjects:
        files += [
            ('Videos_{}.tgz'.format(subject_id),
             'download=1&filepath=Videos&filename=SubjectSpecific_{}.tgz'.format(id)),
        ]

    # out_dir = 'video_download'
    # makedirs(out_dir, exist_ok=True)

    for filename, query in tqdm(files, ascii=True):
        out_file = path.join(out_dir, filename)

        if path.isfile(out_file):
            continue

        if path.isfile(out_file):
            checksum = md5(out_file)
            if checksums.get(out_file, None) == checksum:
                continue

        download_file(BASE_URL + '?' + query, out_file, phpsessid)

# https://stackoverflow.com/a/6718435
def commonprefix(m):
    s1 = min(m)
    s2 = max(m)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1

def extract_tgz(tgz_file, dest):
    # if path.exists(dest):
    #     return
    with tarfile.open(tgz_file, 'r:gz') as tar:
        members = [m for m in tar.getmembers() if m.isreg()]
        member_dirs = [path.dirname(m.name).split(path.sep) for m in members]
        base_path = path.sep.join(commonprefix(member_dirs))
        for m in members:
            m.name = path.relpath(m.name, base_path)
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, dest)

def extract(out_dir,tgzs):
    out_dir = path.join(out_dir,'videos')

    for tgz in tqdm(tgzs,desc='Extracting tgz archives'):
        subject_id = tgz.split('_')[-1].split('.')[0]
        videodir = path.join(out_dir,subject_id)
        makedirs(videodir,exist_ok=True)

        extract_tgz(tgz,videodir)


if __name__ == '__main__':
    config = get_config()
    phpsessid = get_phpsessid(config)
    verify_phpsessid(phpsessid)
    out_dir = config['General']['TARGETDIR']
    download_dir = path.join(out_dir,'video_download')
    makedirs(download_dir,exist_ok=True)
    download_all(phpsessid,out_dir=download_dir)
    tgzs = glob(path.join(download_dir,'*.tgz'))
    extract(out_dir,tgzs)



