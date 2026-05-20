Extracting audio from a video
=============================

.. currentmodule:: birdwatcher

In many cases, extracting audio is as simple as calling the
:meth:`VideoFile.extract_audio` method on a :class:`VideoFile` object:

.. code:: python

    >>> import birdwatcher as bw
    >>> vf = bw.VideoFile('myvideo.mp4')
    >>> vf.extract_audio()
    PosixPath('myvideo.wav')

By default, the resulting file has the same name as the video file, but with
an appropriate audio extension.

If you want to choose the output filename yourself, it is usually best not
to specify the file extension. When possible, Birdwatcher copies the audio
stream directly from the video without re-encoding or resampling it, preserving
the original audio data. This only works with compatible audio container
formats, so Birdwatcher automatically selects the most suitable extension.

For example, AAC-encoded audio is not compatible with the '.wav' container,
so Birdwatcher chooses '.m4a' instead:

.. code:: python

    >>> vf = bw.VideoFile('myvideo_aac.mp4')
    >>> vf.extract_audio(outputpath='mysound')
    PosixPath('mysound.m4a')

Copying the original audio stream is often preferable because it avoids quality
loss and it is typically much faster because no decoding or encoding step is
required. However, some analysis software does not support compressed formats
such as AAC in an '.m4a' container.

In such cases, you can explicitly specify the audio codec to use during
conversion. In general, we recommend 'pcm_f32le' because it preserves audio
information without lossy compression:

.. code:: python

    >>> vf = bw.VideoFile('myvideo_aac.mp4')
    >>> vf.extract_audio(codec='pcm_f32le')
    PosixPath('myvideo_aac.wav')

Birdwatcher selects the '.wav' container here because it is widely compatible
with 32-bit floating-point PCM audio.

Supported audio codecs depend on the installed version of ffmpeg and can
be retrieved with the :func:`supported_audio_codecs` function:

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

If needed, you can explicitly specify both the output filename and the codec,
provided the container format and codec are compatible:

.. code:: python

    >>> import birdwatcher as bw
    >>> vf = bw.VideoFile('myvideo_aac.mp4')
    >>> vf.extract_audio(outputpath='mysound.mkv', codec='pcm_s16le')
    PosixPath('mysound.mkv')
