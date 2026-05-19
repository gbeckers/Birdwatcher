Extracting audio from a video
=============================

In many cases, this is as simple as calling the :meth:`extract_audio` method
on a ``VideoFile`` object:

.. code:: python

    >>> import birdwatcher as bw
    >>> vf = bw.VideoFile('myvideo.mp4')
    >>> vf.extract_audio()
    PosixPath('myvideo.wav')

Birdwatcher used the same file name as the video, but with an audio extension.
If you want to determine the audio file name yourself, it is best in most
cases *not* to specify the file extension. If possible, Birdwatcher tries to
*copy* the audio data from the video to the audio file without any recoding
or resampling so that the audio data remains the same. This will only work
with compatible audio file formats, and Birdwatcher will choose one that
fits best. For example 'aac' encoded audio data is not compatible with a '
.wav' (WAVE) file and will lead to an '.m4a' file:

.. code:: python

    >>> vf = bw.VideoFile('myvideo_aac.mp4')
    >>> vf.extract_audio(outputpath='mysound')
    PosixPath('mysound.m4a')

Copying data instead of recoding is often preferable, but it may not always be
the best solution. For example, an '.m4a' audio file with the same'aac'
encoding is fine for media players, but many sound analysis programs won't read
it. In such cases you can specify the audio codec you want. In the general
case, we recommend 'pcm_f32le' to avoid degradation of information:

.. code:: python

    >>> vf = bw.VideoFile('myvideo_aac.mp4')
    >>> vf.extract_audio(codec='pcm_f32le')
    PosixPath('myvideo_aac.wav')

Birdwatcher chose the '.wav' file format because it is most compatible with
32-bit float PCM encoding.

Supported audio codecs for writing audio files depend on the underlying
'ffmpeg' version and can be retrieved with the :func:`supported_audio_codecs`
function:

.. code:: python

    >>> bw.supported_audio_codecs()
    {'aac', 'ac3', 'ac3_fixed', 'adpcm_adx', 'adpcm_argo', 'adpcm_ima_alp',
    'adpcm_ima_amv', 'adpcm_ima_apm', 'adpcm_ima_qt','adpcm_ima_ssi',
    'adpcm_ima_wav', 'adpcm_ima_ws', 'adpcm_ms', 'adpcm_swf', 'adpcm_yamaha',
    'alac', 'anull', 'aptx', 'aptx_hd', 'comfortnoise', 'dca', 'dfpwm', 'eac3',
    'flac', 'g722', 'g723_1', 'g726', 'g726le', 'libmp3lame', 'libopus',
    'libvorbis', 'mlp', 'mp2', 'mp2fixed', 'nellymoser', 'opus', 'pcm_alaw',
    'pcm_bluray', 'pcm_dvd', 'pcm_f32be', 'pcm_f32le', 'pcm_f64be', 'pcm_f64le',
    'pcm_mulaw', 'pcm_s16be', 'pcm_s16be_planar', 'pcm_s16le', 'pcm_s16le_planar',
    'pcm_s24be', 'pcm_s24daud', 'pcm_s24le', 'pcm_s24le_planar', 'pcm_s32be',
    'pcm_s32le', 'pcm_s32le_planar', 'pcm_s64be', 'pcm_s64le', 'pcm_s8',
    'pcm_s8_planar', 'pcm_u16be', 'pcm_u16le', 'pcm_u24be', 'pcm_u24le',
    'pcm_u32be', 'pcm_u32le', 'pcm_u8', 'pcm_vidc', 'real_144', 'roq_dpcm',
    's302m', 'sbc', 'truehd', 'tta', 'vorbis', 'wavpack', 'wmav1', 'wmav2'}

If you know what you are doing, you can specify both the audio codec *and* the
output extension (which need to be compatible):

.. code:: python

    >>> import birdwatcher as bw
    >>> vf = bw.VideoFile('myvideo_aac.mp4')
    >>> vf.extract_audio(outputpath='mysound.mkv', codec='pcm_s16le')
    PosixPath('mysound.mkv')
