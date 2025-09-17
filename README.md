🧠 Suicide & Depression Detection System

This project is a Flask web application designed to detect stress, depression, or suicidal tendencies using a combination of facial emotion recognition, speech sentiment analysis, and chatbot-based support. It provides a simple interface for users to log in, analyze their state, and receive feedback or guidance.

🚀 Features

👤 User Management – Registration, login, and secure access

📷 Facial Emotion Detection – Uses a CNN model to classify emotions (angry, sad, fear, happy, neutral, etc.) from webcam images

🎙 Speech Emotion Analysis – Captures voice, converts to text, and analyzes for stress-related emotions (sadness, worry, anger)

📝 Combined Stress Detection – Evaluates both facial + speech input to determine user’s stress/depression state

🤖 Chatbot Support – Provides mental health questions & answers based on detected emotions (data from chatbot_qa.csv)

📊 Result Summary – Saves analysis results (result.txt) and provides a final evaluation report

🛠️ Tech Stack

Backend: Flask (Python)

AI/ML: TensorFlow/Keras (Facial Emotion Model), Scikit-learn (Naive Bayes for speech analysis)

Computer Vision: OpenCV (face detection)

Speech Recognition: Google SpeechRecognition API

Database: SQLite (User accounts)

Frontend: HTML, CSS, Bootstrap (templates)

📂 Project Structure
Suicide-Depression-Detection/
│── app.py                  # Main Flask application
│── your_model.keras        # Trained CNN model for facial expressions
│── chatbot_qa.csv          # Emotion-based Q&A dataset
│── tweet_emotions.csv      # Training dataset for speech emotion
│── templates/              # HTML templates (login, signup, chat, results, etc.)
│── uploads/                # Captured images from webcam
│── static/                 # CSS/JS files
│── requirements.txt        # Dependencies
│── README.md               # Documentation

⚙️ Installation & Setup

Clone the repository:

git clone https://github.com/your-username/Suicide-Depression-Detection.git
cd Suicide-Depression-Detection


Install dependencies:

pip install -r requirements.txt


Add the trained model (your_model.keras) inside the project folder.

Run the app:

python app.py


Open in browser:

http://127.0.0.1:5000/

📊 Workflow

User logs in

System captures 3 webcam images → predicts facial emotion

User provides speech input → converts to text → sentiment analyzed

Results are combined → system decides: Stressed / Not Stressed

Chatbot support provided with questions & answers based on emotion

📌 Future Enhancements

Add LSTM/BERT models for more accurate sentiment detection

Cloud deployment for scalability


Integration with emergency helpline APIs for high-risk detection

Real-time chatbot with NLP (Dialogflow/GPT)

![home](https://github.com/latha-shree/Suicide-Depression-Detection-System/blob/main/home.png)

