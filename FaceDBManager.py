from pymongo import MongoClient
import face_recognition
import uuid, os

class MongoDB:
    def __init__(self, db_name="FaceAuthDB", collection_name="Users", host="localhost", port=27017):
        self.client = MongoClient(host, port)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def register_user(self, user_name, image):
        # Generate a random unique ID for the user
        user_id = str(uuid.uuid4())
 
        face_locations = face_recognition.face_locations(image)
        # If no face is found, return
        if len(face_locations) == 0:
            raise ValueError("No face found in the image.")
        if len(face_locations) != 1:
            raise ValueError("Only one face is allowed per image")
        face_encoding = face_recognition.face_encodings(image, face_locations)[0]

        # Insert user data into MongoDB
        user_data = {
            "user_id": user_id,
            "name": user_name,
            "encoding": face_encoding.tolist()
        }
        self.collection.insert_one(user_data)
        print("User registration successful")
        file_path = os.path.join("Temp", 'Temp_image.png')
        os.remove(file_path)
        return user_id

    def authenticate(self, name, image):
        # Load user encodings from MongoDB
        user_data = self.collection.find_one({"name": name})
        if not user_data:
            return False

        # Extract user encodings
        user_encodings = user_data["encoding"]

        # Capture face encoding from the image
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) != 1:
            return False
        unknown_encoding = face_recognition.face_encodings(image, face_locations)[0]
        # Compare unknown encoding with user encodings
        result = face_recognition.compare_faces([user_encodings], unknown_encoding)
        if result[0]:
            return True
        return False
    
    def fetch_all_records(self):
        records = []
        for x in self.collection.find():
            records.append(x)
        return records

    def delete_user(self, name):
        self.collection.delete_one({"name": name})

    def close(self):
        self.client.close()

"""
# Example usage
if __name__ == "__main__":
    auth = FaceDBManager("FaceAuthDB", "Users")

    # Register a user
    user_name = "John Doe"
    user_id = 123
    images = ["images/straight.jpg", "images/left.jpg", "images/right.jpg"]
    auth.register_user(user_name, images)

    # Authenticate the user
    image = "images/test.jpg"
    if auth.authenticate(user_id, image):
        print("User authenticated successfully!")
    else:
        print("Authentication failed!")

    # Delete the user (optional)
    # auth.delete_user(user_id)

    # Close the connection
    auth.close()
"""
