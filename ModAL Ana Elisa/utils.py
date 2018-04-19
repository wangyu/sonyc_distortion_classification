import os
import re
import resampy
import soundfile as psf
import subprocess
import shutil
import decrypt
import tarfile
import tempfile
import librosa
import numpy as np

tar_gz_date_pat = "\d{4}-\d{2}-\d{2}\.tar\.gz"
tar_gz_date_prog = re.compile(tar_gz_date_pat)
recording_tar_gz_pat = "(?P<sensor_id>[a-zA-Z0-9]{2,})[_-](?P<timestamp>[a-zA-Z0-9]{2,}\.\d{1,})\.tar\.gz"
recording_tar_gz_prog = re.compile(recording_tar_gz_pat)
recording_id_pat = "(?P<sensor_id>[a-zA-Z0-9]{2,})[_-](?P<timestamp>[a-zA-Z0-9]{2,}\.\d{1,})"
recording_id_prog = re.compile(recording_id_pat)


def read_audio(filepath, sr=None, mono=True, peak_norm=False):
    """
    Read audio

    Parameters
    ----------
    filepath
    sr
    mono

    Returns
    -------
    y, sr
    """
    try:
        y, _sr = psf.read(filepath)
        y = y.T
    except RuntimeError:
        y, _sr = librosa.load(filepath, mono=False, sr=None)

    if sr is not None and sr != _sr:
        y = resampy.resample(y, _sr, sr, filter='kaiser_fast')
    else:
        sr = _sr

    if mono:
        y = librosa.to_mono(y)

    if peak_norm:
        y /= np.max(np.abs(y))

    return y, sr


def read_encrypted_tar_audio_file(enc_tar_filepath, enc_tar_filebuf=None, sample_rate=44100, **kwargs):
    """
    Given the tarfile (or buffer of a tarfile) of a recording, untar and decrypt

    Parameters
    ----------
    enc_tar_filepath : str
        This is required even if enc_tar_filebuf is given, since we need to extract the name from enc_tar_filepath
    enc_tar_filebuf : File Obj
    sample_rate : int
    kwargs : dict
        `decrypt` arguments

    Returns
    -------
    y : np.array
        Audio data
    sr : int
        Sample rate of `y`
    identifier : str
        Recording identifier
    """
    adir = enc_tar_filepath.replace('.tar.gz', '')

    identifier = os.path.basename(adir)
    mat = recording_id_prog.match(identifier)
    timestamp = mat.group('timestamp')

    if enc_tar_filebuf is None:
        tar = tarfile.open(enc_tar_filepath, mode='r:gz')
    else:
        tar = tarfile.open(fileobj=enc_tar_filebuf, mode='r:gz')
    
    enc = dict()
    for member in tar.getmembers():
        enc[os.path.splitext(member.name)[1]] = tar.extractfile(member)

    # decrypt
    buf = decrypt.decrypt(encrypted_key_filebuf=enc['.key'].read(), encrypted_data_filebuf=enc['.enc'].read(), **kwargs)

    y, sr = psf.read(buf)
    y = resampy.resample(y, sr, sample_rate, filter='kaiser_fast')

    return y, sample_rate, identifier


def read_encrypted_tar_audio_file_from_day_tar(day_tar_filepath, enc_tar_filename, sample_rate, **kwargs):
    """
    Extract and decrypt a single file from a day's tar file, all using buffers

    Parameters
    ----------
    day_tar_filepath : str
    enc_tar_filename : str
    sample_rate : int
    kwargs : dict
        `decrypt` arguments

    Returns
    -------
    y : np.array
        Audio data
    sr : int
        Sample rate
    identifier : str
        Recording identifier
    """
    # open tar file
    tar = tarfile.open(day_tar_filepath)

    # get tar member info
    member = tar.getmember(enc_tar_filename)

    # extract recording file from day file
    enc_tar_filebuf = tar.extractfile(member)

    return read_encrypted_tar_audio_file(enc_tar_filename,
                                         enc_tar_filebuf=enc_tar_filebuf,
                                         sample_rate=sample_rate,
                                         **kwargs)


def find_files_in_dirs(dirs, extensions=('.wav', '.mp3', '.aif', '.aiff', '.flac'), depth=None):
    """
    Find all files in the directories `dir` and their subdirectories with `extensions`, and return the full file path

    Parameters
    ----------
    dirs : list[str]
    extensions : list[str]

    Yields
    -------
    afile : str
    """
    for adir in dirs:
        if depth is not None:
            _found = walklevel(adir, depth)
        else:
            _found = os.walk(adir)
        for root, subdirs, files in _found:
            for f in files:
                if os.path.splitext(f)[1].lower() in extensions:
                    yield os.path.join(root, f)


def find_files_in_dir(adir, extensions=('.wav', '.mp3', '.aif', '.aiff', '.flac')):
    """
    Find all files in directory `dir` with `extensions`, and return the full file path

    Parameters
    ----------
    dirs : list[str]
    extensions : list[str]

    Returns
    -------
    files : list[str]
    """
    return [os.path.join(adir, afile) for afile in os.listdir(adir) if os.path.splitext(afile)[1].lower() in extensions]


def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


def r128stats(filepath, ffmpeg_path='ffmpeg'):
    """ takes a path to an audio file, returns a dict with the loudness
    stats computed by the ffmpeg ebur128 filter """
    ffargs = [ffmpeg_path,
              '-nostats',
              '-i',
              filepath,
              '-filter_complex',
              'ebur128',
              '-f',
              'null',
              '-']
    try:
        proc = subprocess.Popen(ffargs, stderr=subprocess.PIPE,
                                universal_newlines=True)
        stats = proc.communicate()[1]
        summary_index = stats.rfind('Summary:')
        summary_list = stats[summary_index:].split()
        i_lufs = float(summary_list[summary_list.index('I:') + 1])
        i_thresh = float(summary_list[summary_list.index('I:') + 4])
        lra = float(summary_list[summary_list.index('LRA:') + 1])
        lra_thresh = float(summary_list[summary_list.index('LRA:') + 4])
        lra_low = float(summary_list[summary_list.index('low:') + 1])
        lra_high = float(summary_list[summary_list.index('high:') + 1])
        stats_dict = {'I': i_lufs, 'I Threshold': i_thresh, 'LRA': lra,
                      'LRA Threshold': lra_thresh, 'LRA Low': lra_low,
                      'LRA High': lra_high}
    except:
        return False
    return stats_dict


def get_integrated_lufs(filepath, extend=True, peak_normalize_first=False, ffmpeg_path='ffmpeg'):
    '''Returns the integrated lufs for an audiofile'''
    # check length, make sure at least 1 sec
    x, fs = read_audio(filepath, peak_norm=peak_normalize_first, mono=True)

    if x.shape[0] <= fs and extend:
        x = np.pad(x, pad_width=((0, fs),), mode='constant', constant_values=(0,))

    tf = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)

    psf.write(tf, x, fs)
    tf.close()

    loudness_stats = r128stats(tf.name, ffmpeg_path=ffmpeg_path)

    os.remove(tf.name)

    if not loudness_stats:
        raise Exception(
            'Unable to obtain LUFS state for {:s}'.format(filepath))
    return loudness_stats['I']


def linear_gain(i_lufs, goal_lufs=-23):
    """
    takes a floating point value for i_lufs, returns the necessary
    multiplier for audio gain to get to the goal_lufs value
    """
    gain_log = -(i_lufs - goal_lufs)
    return 10 ** (gain_log / 20.0)


def ff_apply_gain(inpath, outpath, linear_amount, ffmpeg_path='ffmpeg'):
    """
    creates a file from inpath at outpath, applying a filter
    for audio volume, multiplying by linearAmount
    """
    tf = tempfile.NamedTemporaryFile(suffix=os.path.splitext(outpath)[1], delete=False)

    ffargs = [ffmpeg_path, '-y', '-i', inpath,
              '-af', 'volume=' + str(linear_amount),]
    if os.path.splitext(outpath)[1].lower() == '.mp3':
        ffargs += ['-acodec', 'libmp3lame', '-aq', '0']
    ffargs += [tf.name]
    try:
        subprocess.call(ffargs)
    except:
        return False

    shutil.move(tf.name, outpath)

    return True


def normalize_audio_lufs(infile, outfile, target_lufs=-23, ffmpeg_path='ffmpeg'):
    """
    Take infile, normalize to goal_lufs, and save to outfile.
    Parameters
    ----------
    infile : str
        Path to input audio file
    outfile : str
        Path to output audio file
    goal_lufs : float
        Desired LUFS level after normalization

    Returns
    -------
    new_lufs
    """

    current_lufs = get_integrated_lufs(infile, extend=True, ffmpeg_path=ffmpeg_path)
    if not current_lufs:
        raise Exception(
            'Unable to obtain LUFS state for {:s}'.format(infile))

    gain_amount = linear_gain(current_lufs, goal_lufs=target_lufs)
    ff_gain_success = ff_apply_gain(infile, outfile, gain_amount, ffmpeg_path=ffmpeg_path)
    if not ff_gain_success:
        raise Exception(
            'Unable to apply gain of {:.2f} (goal LUFS={:.2f}) for input={:s} '
            'output={:s}'.format(gain_amount, target_lufs, infile, outfile))

    return get_integrated_lufs(outfile, extend=True, ffmpeg_path=ffmpeg_path)


def lufs_normalize(directory=None, file_list=None, suffix=None, target_lufs=None, output_ext=None, ffmpeg_path='ffmpeg'):
    """
    This utility performs lufs normalization on a directory or list of files.

    Parameters
    ----------
    directory : str
        Input directory of audio files to process. Either this or `file_list` must be defined. Default is None.
    file_list : list of str
        List of audio files to process. Either this or `directory` must be defined. Default is None.
    suffix : str
        The suffix to append to the output filenames. If `None`, then the input files will be overwritten. Default is
        None.
    target_lufs : float
        The target LUFS to which we normalize. If `None`, then calculate the minimum LUFS of the peak normalized files
        and normalize to that.

    Returns
    -------
    output_file_list : list of str
    pre_norm_values : list of float
    post_norm_values : list of float
    """
    if file_list is None:
        if directory is None:
            raise Exception('Arguments `file_list` or `directory` must be defined')
        file_list = []

        for path, _dirs, files in os.walk(directory):
            file_list.extend([os.path.join(path, f) for f in files if (os.path.splitext(f)[1] == ".wav" or
                                                                       os.path.splitext(f)[1] == ".WAV" or
                                                                       os.path.splitext(f)[1] == ".AIFF" or
                                                                       os.path.splitext(f)[1] == ".aiff" or
                                                                       os.path.splitext(f)[1] == ".aif" or
                                                                       os.path.splitext(f)[1] == ".AIF")])
    pre_norm_values = np.asarray([get_integrated_lufs(file_name, peak_normalize_first=True, ffmpeg_path=ffmpeg_path) for file_name in file_list])

    if target_lufs is None:
        target_lufs = min(pre_norm_values[pre_norm_values.nonzero()])

    print('target: {}'.format(target_lufs))

    if suffix is not None:
        output_file_list = []
        for f in file_list:
            head, tail = os.path.splitext(f)
            if output_ext is not None:
                tail = output_ext
            output_file_list.append(head + suffix + tail)
    else:
        output_file_list = file_list

    post_norm_values = np.asarray([normalize_audio_lufs(file_list[i],
                                                        output_file_list[i],
                                                        target_lufs,
                                                        ffmpeg_path=ffmpeg_path) for i in range(len(file_list))])
    print('Normalization complete.')
    print(pre_norm_values, post_norm_values)
    return file_list, pre_norm_values, post_norm_values