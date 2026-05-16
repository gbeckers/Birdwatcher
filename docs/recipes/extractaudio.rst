Extracting audio from a video
=============================

In many cases this is simple:

.. code:: python

    >>> import birdwatcher as bw
    >>> vf = bw.VideoFile('myvideo.mp4')
    >>> vf.extract_audio()
    PosixPath('myvideo.wav')

Birdwatcher wrote a WAVE audio file with the same name but, in this case,
with a '.wav' extension. If possible, it tries to *copy* the audio data from
the video to the audio file. That is, without transforming the audio data. This
is often preferable but it may not be always what you want. For example, if the
audio in the video file is compressed using the 'aac' codec, copying that data
would lead to an an audio file with aac compression, which is fine for media
players, but many sound analysis programs won't read it. In such cases you can
specify the audio codec you want. We recommend 'pcm_f32le':

.. code:: python

    >>> vf = bw.VideoFile('myvideo_aac.mp4')
    >>> vf.extract_audio(codec='pcm_f32le')
    PosixPath('myvideo_aac.wav')

If you want to determine the audio file name yourself, it is best in most
cases *not* to specify the file extension. Birdwatcher will choose one that
fits best with the audio codec used in de video file (which is copied by
default):

.. code:: python

    >>> vf = bw.VideoFile('myvideo_aac.mp4')
    >>> vf.extract_audio(outputpath='mysound')
    PosixPath('mysound.m4a')

An overview of the supported audio codecs for writing audio files:

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

If you know what you are doing, you can specify both the audio codec and the
output extension (which need to be compatible):

.. code:: python

    >>> import birdwatcher as bw
    >>> vf = bw.VideoFile('myvideo_aac.mp4')
    >>> vf.extract_audio(outputpath='mysound.mkv', codec='pcm_s16le')
    PosixPath('mysound.mkv')
