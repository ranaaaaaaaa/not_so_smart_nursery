from tkinter import *
from pyvidplayer2 import VideoTkinter

def play(window):
    video_player = Toplevel(window)
    video_player.title("Video Player")
    video_player.attributes("-topmost", True)
    video_player.state("zoomed")

    canvas = Canvas(video_player)
    canvas.pack(fill="both", expand=True)

    video = VideoTkinter("calma.mp4")
    video.play()

    def update():
        if video.active:
            video.draw(canvas, (0,0))
            video_player.after(33, update)   # ~30 fps
        else:
            video.close()
            video_player.destroy()

    video_player.protocol("WM_DELETE_WINDOW", lambda: (video.close(), video_player.destroy()))
    update()

    return video_player

# .draw(), .play(), .close(), .active, .seek(), .pause()	video (the player)
# .after(), .destroy(), .protocol(), .title(), .attributes(), .geometry()	video_player (the window)
