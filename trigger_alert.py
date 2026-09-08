from tkinter import *
from playsound3 import playsound
from telegram_bot import send_alert

def trigger_alert(window):
    alert = Toplevel(window)
    alert.attributes("-topmost", True)

    alert.title("Serious Alert!")
    alert.config(bg="red")

    alert_label = Label(
        alert,
        text="Smoke/Gas Alert!",
        font=("ALGERIAN", 15, "bold"),
        bg="red",
        fg="white"
    )
    alert_label.pack()

    telegram_label = Label(
        alert,
        text="A TELEGRAM NOTIF WAS SENT TO YOUR DEVICE!",
        font=("ALGERIAN", 15, "bold"),
        bg="red",
        fg="#78d4ff"
    )
    telegram_label.pack()

    danger = PhotoImage(file='5atar.png')
    danger_label = Label(
        alert,
        font=("ALGERIAN", 15, "bold"),
        image=danger,
        fg="#78d4ff",
    )
    danger_label.pack()
    danger_label.image = danger
    danger_label.pack(pady=20)
    playsound("mashkal.mp3", block=False)

    send_alert()