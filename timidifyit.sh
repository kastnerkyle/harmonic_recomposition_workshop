file=$1
bn=${file%.mid}
timidity --output-24bit -Ow $file -o $bn.wav
ffmpeg -i $bn.wav -acodec pcm_s16le -ar 44100 $bn_1.wav
mv $bn_1.wav $bn.wav
lame $bn.wav o.mp3
mv o.mp3 $bn.mp3
