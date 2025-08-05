import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pywt
import soundfile as sf
from scipy.signal import stft
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AudioAnalyzerApp:
    def __init__(self, master):
        self.master = master
        master.title("Audio Signal Analyzer")

        self.load_button = tk.Button(master, text="Insert audio file", command=self.load_audio)
        self.load_button.pack()

        self.plot_frame = tk.Frame(master)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

    def load_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        if not file_path:
            return

        
        y, sr = librosa.load(file_path, sr=None, mono=True)
        self.plot_audio_analysis(y, sr)

    def plot_audio_analysis(self, y, sr):
        
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, axs = plt.subplots(4, 1, figsize=(8, 10))
        fig.tight_layout(pad=3.0)

        
        t = np.linspace(0, len(y)/sr, num=len(y))
        axs[0].plot(t, y)
        axs[0].set_title("Amplitude ande time")
        axs[0].set_xlabel("Time [s]")
        axs[0].set_ylabel("Amplitude")

        
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axs[1])
        axs[1].set_title("Spectrogram (STFT)")
        fig.colorbar(img, ax=axs[1], format="%+2.0f dB")

        
        coeffs = pywt.wavedec(y, 'db4', level=4)
        for i, c in enumerate(coeffs):
            axs[2].plot(c, label=f'Level {i}')
        axs[2].set_title("Wavelet Transform (db4)")
        axs[2].legend(loc='upper right')

        
        f, t_spec, Zxx = stft(y, fs=sr)
        axs[3].pcolormesh(t_spec, f, np.abs(Zxx), shading='gouraud')
        axs[3].set_title("STFT")
        axs[3].set_ylabel("Frequency [Hz]")
        axs[3].set_xlabel("Time [s]")

        
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


root = tk.Tk()
app = AudioAnalyzerApp(root)
root.mainloop()
