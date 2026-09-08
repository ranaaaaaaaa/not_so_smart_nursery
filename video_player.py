from tkinter import *
from tkintervideo import player
# install ffmpeg
def play():
    video = Toplevel()
    video.attributes("-topmost", True)
    video.title("Video Player")

    video_player = player.Player(video)
    video_player._current_frame_size = (400, 400)
    video_player.pack()

    video_player.load("calma.mp4")
    video_player.play()