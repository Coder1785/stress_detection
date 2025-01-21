import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd
from pydub import AudioSegment
import librosa
import numpy as np
from tkinter import PhotoImage
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class StressDetectionApp:

    def __init__(self, master):
        self.master = master
        master.title("Stress Detection App")
        master.geometry("1200x600")
        master.configure(bg='#333333')
        self.audio_file_path = None

        self.user_name = "" 
        self.age = None
        self.gender = ""
        self.city = ""
        self.area = ""
        self.degree = ""
        self.sem = None

        self.pred = None

        main_frame = tk.Frame(master, bg='#333333')
        main_frame.pack(fill='both', expand=True)

        # User Data Frame (left side)
        user_frame = tk.Frame(main_frame, bg='#333333')
        user_frame.pack(side='left', fill='y', padx=1, pady=20, expand=True)

        user_box = tk.LabelFrame(user_frame, bg='#333333', padx=40, pady=40)
        user_box.pack(fill='both', expand=True)
        user_label = tk.Label(user_box, text="User Data", font=('Lobster', 20, 'bold'), bg='#333333', fg='#FF3399')
        user_label.pack()

        labels_box = tk.LabelFrame(user_frame, bg='#333333', padx=50, pady=50)
        labels_box.pack(pady=1, fill='both', expand=True)
        self.labels = ['Name:', 'Age:', 'Gender:', 'City of Residence:', 'Area of Residence:', 'Area of Study:', 'Current Semester:']
        self.examples = ['e.g., Ahmad Ali', 'e.g., 30', 'e.g., M for Male and F for Female', 'e.g., Lahore', 'e.g., Cantt', 'e.g., Computer Science', 'e.g., Spring 2024']
        self.entries = []
        for i, label_text in enumerate(self.labels):
            frame = tk.Frame(labels_box, bg='#333333')
            frame.pack(fill='x')
            label = tk.Label(frame, text=label_text, font=('Lobster', 14), fg='#FF3399', bg='#333333', width=15, anchor='e')  
            label.pack(side='left', padx=(5, 5), pady=5)
            entry = tk.Entry(frame, width=50, fg='#666666')
            entry.insert(0, self.examples[i]) 
            entry.bind("<FocusIn>", lambda event, entry=entry, index=i: self.on_entry_focus_in(event, entry, index))  
            entry.bind("<FocusOut>", lambda event, entry=entry, index=i: self.on_entry_focus_out(event, entry, index))  
            entry.pack(side='right', fill='x', expand=True, padx=(0, 20), pady=5)
            self.entries.append(entry)

        # Actions Frame (right side)
        actions_frame = tk.Frame(main_frame, bg='#333333')
        actions_frame.pack(side='right', fill='y', padx=1, pady=20, expand=True)

        browse_box = tk.LabelFrame(actions_frame, bg='#333333', padx=20, pady=1)
        browse_box.pack(pady=20, fill='both', expand=True)
        browse_button = tk.Button(browse_box, text="Browse Audio File", font=('Lobster', 14, 'bold'), fg='#FF3399', bg='#444444', command=self.select_audio_file, width=20)
        browse_button.pack(pady=100, padx=1)

        detect_box = tk.LabelFrame(actions_frame, bg='#333333', padx=20, pady=10)
        detect_box.pack(pady=20, fill='both', expand=True)
        detect_button = tk.Button(detect_box, text="Detect Stress", font=('Lobster', 30, 'bold'), fg='#FF3399', bg='#444444', command=self.detect_stress, width=20)
        detect_button.pack(pady=50, padx=1)

    def on_entry_focus_in(self, event, entry, index):
        if entry.get() == self.examples[index]:
            entry.delete(0, tk.END)
            entry.config(fg='#000000')  #change text color when writing

    def on_entry_focus_out(self, event, entry, index):
        if not entry.get():
            entry.insert(0, self.examples[index])
            entry.config(fg='#666666')  #change text color back to placeholder color
        else:
            self.update_data(index, entry.get())

    def update_data(self, index, value):
        if index == 0: 
            self.user_name = value
        elif index == 1: 
            self.age = value 
        elif index == 2: 
            self.gender = value 
        elif index == 3: 
            self.city = value 
        elif index == 4: 
            self.area = value  
        elif index == 5: 
            self.degree = value 
        elif index == 6: 
            self.sem = value  

            # Check if all required fields are filled
            if self.user_name and self.age and self.gender and self.city and self.area and self.degree and self.sem:
                self.save_to_excel()

    def save_to_excel(self):
        path = r"D:\fyp_data\data.xlsx"
        try:
            df = pd.read_excel(path)
            print("Excel file read successfully")
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            df = pd.DataFrame()

        new_data = pd.DataFrame([{
            "Name": self.user_name,
            "City": self.city,
            "Area of Residence": self.area,
            "Discipline": self.degree,
            "Current Semester": self.sem,
            "Gender": self.gender,
            "Age": self.age,
        }])

        df = pd.concat([df, new_data], ignore_index=True)
        
        try:
            df.to_excel(path, index=False)
            print("Data saved to Excel file successfully")
        except Exception as e:
            print(f"Error saving to Excel file: {e}")
                

    @staticmethod
    def convert_to_wav(input_file, output_path):
        input_format = input_file.split('.')[-1].lower()
        if input_format not in ['mp3', 'mp4', 'm4a']:
            raise ValueError("Unsupported input file format. Supported formats: MP3, MP4, M4A.")
        input_file_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_path, f"{input_file_name}.wav")
        audio = AudioSegment.from_file(input_file, format=input_format)
        audio.export(output_file, format="wav")
        return output_file

    @staticmethod
    def extract_audio_features(audio_file_path):
        # Your extract_audio_features code here
        y, sr = librosa.load(audio_file_path)

        zero_crossings_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
        rms_energy = np.mean(librosa.feature.rms(y=y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y), axis=1).tolist()
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y))
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1).tolist()
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1).tolist()
        hnr = np.mean(librosa.effects.harmonic(y=y))
        tonnetz = np.mean(librosa.feature.tonnetz(y=y, sr=sr), axis=1).tolist()

        audio_features = {
            'audio_file': os.path.basename(audio_file_path),
            'zero_crossings_rate': zero_crossings_rate,
            'rms_energy': rms_energy,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_contrast': spectral_contrast,
            'spectral_rolloff': spectral_rolloff,
            'mfccs': mfccs,
            'tempo': tempo,
            'chroma': chroma,
            'hnr': hnr,
            'tonnetz': tonnetz
        }

        return audio_features

    @staticmethod
    def process_wav_folder(file):
        # Your process_wav_folder code here
        audio_features_list = []
        if file.endswith('.wav'):
            file_name = os.path.basename(file)
            audio_features = StressDetectionApp.extract_audio_features(file)
            audio_features_list.append(audio_features)

        audio_features_df = pd.DataFrame(audio_features_list)
        return audio_features_df
    
    @staticmethod
    def expanding_col(audio_features_df):
        # Your expanding_col code here
        list_columns = ['chroma', 'mfccs', 'spectral_contrast', 'tonnetz']
        max_len = [12, 13, 7, 6]
        for idx, col in enumerate(list_columns):
            for i in range(max_len[idx]):
                new_col_name = f'{col}_{i+1}'

                if audio_features_df[col].apply(lambda x: isinstance(x, list)).all():
                    audio_features_df[new_col_name] = audio_features_df[col].apply(lambda x: x[i] if len(x) > i else None)
                else:
                    audio_features_df[new_col_name] = audio_features_df[col]

        audio_features_df.drop(list_columns, axis=1, inplace=True)
        return audio_features_df

    @staticmethod
    def apply_model(audio_features_df):
        mean_values = np.array([ 
            6.60019084e-02,  5.70621330e-02,  1.64890694e+03,  1.99001202e+03,
            3.26314225e+03,  1.24944607e+02,  2.53006500e-08,  3.67719645e-01,
            3.46299008e-01,  3.24168293e-01,  3.04794669e-01,  2.86613489e-01,
            2.79719900e-01,  2.86229858e-01,  3.18716431e-01,  3.66157456e-01,
            3.99378544e-01,  4.06757321e-01,  3.90697506e-01, -3.07131018e+02,
            1.16965165e+02,  3.25590153e+00,  3.07789080e+01,  9.85400818e+00,
            -4.19811755e+00, -8.01485528e+00, -7.62795324e+00, -8.67456450e+00,
            -5.94534782e-01, -7.51456976e+00, -1.50422012e-01, -4.98458722e+00,
            2.31648102e+01,  1.67812043e+01,  1.95365055e+01,  1.69586198e+01,
            1.82509007e+01,  2.10418893e+01,  4.92887691e+01, -5.58390972e-03,
            2.91237505e-03, -1.11287003e-02,  6.37072529e-03,  1.74824732e-03,
            -7.31193354e-04
        ])
        std_dev_values = np.array([
            1.72074979e-02, 3.15701035e-02, 2.67530412e+02, 1.79398990e+02,
            5.36065307e+02, 1.55688370e+01, 4.49246414e-07, 1.10108111e-01,
            1.04994046e-01, 9.88507136e-02, 9.43013583e-02, 8.49429013e-02,
            8.34932348e-02, 8.00582078e-02, 9.59225333e-02, 1.21353217e-01,
            1.27394701e-01, 1.25605169e-01, 1.16431425e-01, 5.76667551e+01,
            1.63803164e+01, 1.24869054e+01, 8.61531518e+00, 9.55339164e+00,
            8.66620223e+00, 7.74972846e+00, 7.19102972e+00, 7.54577129e+00,
            6.30862752e+00, 4.93039750e+00, 5.28719547e+00, 5.57541623e+00,
            2.98673012e+00, 1.82730080e+00, 1.59126521e+00, 1.36078281e+00,
            1.30281885e+00, 2.55825047e+00, 1.44860006e+00, 2.26797996e-02,
            2.09250133e-02, 4.44592483e-02, 4.52308625e-02, 1.64252618e-02,
            1.30545615e-02
        ])

        X = audio_features_df.drop(['audio_file'], axis=1).values
        X_normalized = (X - mean_values) / std_dev_values 
        X_reshaped = X_normalized.reshape(X_normalized.shape[0], X_normalized.shape[1], 1)
        print(X_reshaped)
        print(X_reshaped.shape)
        
        loaded_model = load_model(r"D:\fyp_data\project\best_model.h5")
        predictions = loaded_model.predict(X_reshaped)
        print(predictions)
        predicted_class_index = np.argmax(predictions, axis=1)
        return predictions, predicted_class_index

    def select_audio_file(self):
        self.audio_file_path = filedialog.askopenfilename(initialdir='/', title="Select Audio File",
                                                     filetypes=(("Audio Files", " *.mp3 *.mp4 *.m4a"), ("All Files", "*.*")))
        if self.audio_file_path:
            print("Selected audio file:", self.audio_file_path)


    def detect_stress(self):
        if not self.audio_file_path:
            error_window = tk.Toplevel(self.master)
            error_window.title("Error")
            error_window.geometry("300x100")
            error_window.configure(bg='#333333')
            error_label = tk.Label(error_window, text="Error!! please select an audio file.", font=('Lobster', 12, 'bold'), fg='red', bg='#333333')
            error_label.pack(pady=20)
            return 

        if any(entry.get() == example for entry, example in zip(self.entries, self.examples)):
            error_window = tk.Toplevel(self.master)
            error_window.title("Error")
            error_window.geometry("300x100")
            error_window.configure(bg='#333333')
            error_label = tk.Label(error_window, text="Error!! please fill in all input fields.", font=('Lobster', 12, 'bold'), fg='red', bg='#333333')
            error_label.pack(pady=20)
            return  

        try:
            output_path = r"D:\fyp_data\wv"  # Change this to your desired output folder
            wav_path = StressDetectionApp.convert_to_wav(self.audio_file_path, output_path)
            audio_features_df = StressDetectionApp.process_wav_folder(wav_path)
            audio_features_df = StressDetectionApp.expanding_col(audio_features_df)
            pred,predicted_class_index = StressDetectionApp.apply_model(audio_features_df)

            self.show_result_popup(pred, predicted_class_index)
        except Exception as e:
            print("Error:", e)

    def show_result_popup(self, pred, predicted_class_index):
        result_window = tk.Toplevel(self.master)
        result_window.title("Stress Detection App Result")
        result_window.geometry("800x500")
        result_window.configure(bg='#333333')

        # Text above the message label
        user_box = tk.LabelFrame(result_window, bg='#333333', padx=5, pady=1)
        user_box.pack(fill='both', expand=True)
        above_text = tk.Label(user_box, text="Stress Detection Analysis Result", font=('Arial', 18, 'bold'), bg='#333333', fg='#FF3399')
        above_text.pack()

        if predicted_class_index == 0:
            message_label = f"Hello {self.user_name}, I have good news for you! Based on the analysis of your voice, there are no indications of stress symptoms. Your voice reflects a balanced and calm state of mind."
        elif predicted_class_index == 1:
            message_label = f"Hello {self.user_name}, I'm sorry to tell you that your voice analysis suggests signs of stress. Your vocal pitch is notably high, which can be an indicator of heightened stress levels. It's important to take steps to manage and alleviate stress."
        elif predicted_class_index == 2:
            message_label = f"Hello {self.user_name}, I'm sorry to inform you that your voice analysis indicates signs of stress. Your vocal pitch is low and slow, which can be associated with feelings of fatigue or low energy. It's essential to address these feelings and seek support if needed."
        elif predicted_class_index == 3:
            message_label = f"Hello {self.user_name}, I regret to inform you that your voice analysis suggests signs of stress. Your vocal pitch is low, which might indicate a sense of sadness or emotional strain. It's important to prioritize self-care and seek assistance if you're feeling overwhelmed."
        elif predicted_class_index == 4:
            message_label = f"Hello {self.user_name}, I'm sorry to inform you that your voice analysis indicates signs of stress. There is a noticeable sense of irritation in your voice, which can be a manifestation of underlying stressors. It's crucial to identify and address the sources of irritation for your well-being."
        else:
            message_label = f"Hello {self.user_name}, I'm sorry to inform you that your voice analysis suggests signs of stress. There is a significant amount of stuttering in your voice, which can be a reflection of anxiety or tension. Seeking professional support can help in managing these challenges."

        # Box around the message label
        user_box = tk.LabelFrame(result_window, bg='#333333', padx=5, pady=1)
        user_box.pack(fill='both', expand=True)
        label = tk.Label(user_box, text=message_label, font=('Lobster', 14, 'bold'), fg='#FF3399', bg='#333333', wraplength=500)
        label.pack()

        # Text below the message label
        below_text = tk.Label(result_window, text="For further assistance, please consult a mental health professional.", font=('Arial', 12, 'bold'),  bg='#333333', fg='#FF3399')
        below_text.pack(pady=10)


        # label = tk.Label(result_window, text=pred, font=('Lobster', 13, 'bold'), fg='#FF3399', bg='#333333')
        # label.pack(pady=20, padx=20)

       # path = r"D:\fyp_data\data.xlsx"
        #df = pd.read_excel(path)
        # Display count of men and women
        #count_gender(df, result_window)

        # Display count of ages
        #count_age(df, result_window)

        # Display count of discipline
        #count_discipline(df, result_window)

        #close_button = tk.Button(result_window, text="Close", command=result_window.destroy)
        #close_button.pack(pady=10)

        result_window.protocol("WM_DELETE_WINDOW", lambda: self.close_windows(result_window))

    def close_windows(self, window):
        window.destroy()
        self.master.destroy()

def create_graph(df, parent, title, x_label, y_label):
    fig, ax = plt.subplots(figsize=(4, 7))
    df.plot(kind='bar', ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Display the count of each bar
    for rect in ax.patches:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 1, f'{height}', ha='center', va='bottom', fontsize=10, color='black')

    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    #canvas.get_tk_widget().pack(pady=20)
    canvas.get_tk_widget().pack(side=tk.LEFT, padx=10, pady=10)  # Adjust packing

def count_gender(df, parent):
    gender_counts = df['Gender'].value_counts()
    create_graph(gender_counts, parent, 'Count of Men and Women', 'Gender', 'Count')

def count_age(df, parent):
    age_counts = df['Age'].value_counts()
    create_graph(age_counts, parent, 'Count of Ages', 'Age', 'Count')

def count_discipline(df, parent):
    discipline_counts = df['Discipline'].value_counts()
    create_graph(discipline_counts, parent, 'Count of Discipline', 'Discipline', 'Count') 


# main
root = tk.Tk()
app = StressDetectionApp(root)

result_label = tk.Label(root, text="", font=('Lobster', 12), fg='#FF3399', bg='#333333')
result_label.pack()

root.mainloop()
