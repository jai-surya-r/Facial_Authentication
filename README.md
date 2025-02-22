# Project Overview
SecureFaces is a facial authentication system that enhances cybersecurity by incorporating a Deepfake detection feature. The system utilizes two models, MesoNet and ResNetLSTM, for authentication based on user requirements. If speed is the priority, MesoNet is used, while if accuracy is important, ResNetLSTM is employed.

The system stores users' facial data in MongoDB for authentication purposes.

# Deepfakes
Deepfakes are manipulated media that use artificial intelligence and machine learning techniques to create fake videos or images, often depicting people saying or doing things they never did. These fake videos pose a significant threat to cybersecurity and can be used for various malicious purposes, including spreading disinformation, blackmail, and impersonation.

# Deepfake images
SecureFaces is a facial authentication system designed to combat the threat of Deepfakes. By incorporating advanced deepfake detection features alongside traditional facial recognition, SecureFaces provides robust security against impersonation attacks and ensures the authenticity of user identities.

# User Workflow
- Signup: Users create an account by providing a username and capturing a picture of their face. The face image is processed using the face_recognition module, and the user's username and face encodings are stored in MongoDB.
- Signin: Users choose the authentication model (MesoNet or ResNetLSTM) and upload a video of themselves for authentication. The video is processed by the face_recognition module, comparing the face encodings with those stored in the database. Additionally, the video is analyzed for deepfake indicators using the chosen model.
- If the face encodings match and no deepfake is detected, the user is successfully authenticated and granted access.
- If a deepfake is detected, the login attempt is denied, and appropriate security measures are taken.

# Available Models
SecureFaces offers two models for authentication:

MesoNet: A model optimized for speed.
ResNetLSTM: A model optimized for accuracy.
Users can choose the model based on their specific requirements.

# MesoNet
MesoNet is an image-based facial recognition system known for its speed and efficiency. It achieves an accuracy rate of 89.6%, making it suitable for applications where quick authentication is essential.
The architecture of MesoNet consists of a lightweight convolutional neural network (CNN) that extracts features from facial images. These features are then passed through a series of layers for classification. Despite its simplicity, MesoNet demonstrates impressive performance in detecting and authenticating faces.

Additional features of MesoNet:
Low computational complexity
Real-time processing capabilities
Robustness against various facial expressions and lighting conditions

# ResNetLSTM
ResNetLSTM is a video-based facial recognition system that offers high accuracy rates across three categories: 20 frames accuracy: 90%, 40 frames accuracy: 95%, and 60 frames accuracy: 97%. This model is designed to provide superior authentication accuracy, making it suitable for applications where security is paramount.
The architecture of ResNetLSTM combines the power of residual neural networks (ResNet) with long short-term memory (LSTM) networks. ResNet is used for feature extraction from video frames, while LSTM captures temporal dependencies in facial motion over time. This combination allows ResNetLSTM to effectively distinguish genuine faces from deepfake or manipulated videos.

Additional features of ResNetLSTM:
Adaptive learning capabilities to detect evolving deepfake techniques
Highly accurate authentication even in challenging scenarios
Scalability to process videos of varying lengths
