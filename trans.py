import soundfile as sf
import librosa


def convert_mp3_to_wav(mp3_file, wav_file):
    # 使用librosa加载MP3文件
    audio, sr = librosa.load(mp3_file, sr=None)

    # 将音频数据保存为WAV文件
    sf.write(wav_file, audio, sr)


# 示例用法
mp3_file = 'aaa.mp3'  # 输入的MP3文件路径
wav_file = 'output.wav'  # 转换后的WAV文件保存路径

# 调用函数进行转换和保存
convert_mp3_to_wav(mp3_file, wav_file)