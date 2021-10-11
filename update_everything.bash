set -e

python3 generate_image.py 1920 1080
python3 generate_image.py 2560 1600
python3 generate_image.py 3440 1440
python3 generate_image.py 3840 2160

python3 generate_video.py 1920 1080
python3 generate_video.py 2560 1600
python3 generate_video.py 3440 1440
python3 generate_video.py 3840 2160
