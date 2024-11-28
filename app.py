from flask import Flask, request, jsonify, render_template
import psycopg2
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
import pandas as pd

# Database Connection Parameters
db_params = {
    'user': 'postgres',
    'password': 'SQLanshu728',
    'host': 'localhost',
    'port': '5432',
    'database': 'postgres'
}

# Flask App Initialization
app = Flask(__name__)

# Chatbot Class
class CollegeChatbot:
    def __init__(self, db_params):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        try:
            connection_string = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
            self.engine = create_engine(connection_string)
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            print("Database connection successful!")
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            raise

    def embed_query(self, query):
        try:
            embedding = self.model.encode(query)
            print(f"Query Embedding: {embedding}")
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise

    def find_similar_questions(self, query_embedding):
        try:
            with self.engine.connect() as connection:
                sql_query = text(""" 
                    SELECT id, question, answer, 
                        (embedding <=> CAST(:query_embedding AS vector)) AS cosine_distance
                    FROM qa1_embeddings
                    ORDER BY cosine_distance
                    LIMIT 5;
                """)
                result = connection.execute(sql_query, {'query_embedding': query_embedding.tolist()})
                result_df = pd.DataFrame(result.fetchall(), columns=['id', 'question', 'answer', 'cosine_distance'])
                print(f"Query Results: {result_df}")
                return result_df
        except Exception as e:
            print(f"Error finding similar questions: {e}")
            raise

    def get_answer(self, user_query, similarity_threshold=0.5):
        try:
            # Embed the user query to compare it with the stored questions
            query_embedding = self.embed_query(user_query)
            
            # Find similar questions in the database
            similar_questions = self.find_similar_questions(query_embedding)
            
            # Check if there are any similar questions found
            if not similar_questions.empty:
                top_match = similar_questions.iloc[0]  # Take the top matching question
                similarity_score = 1 - top_match['cosine_distance']  # Calculate similarity score
                
                # If similarity score is above the threshold, return the answer
                if similarity_score >= similarity_threshold:
                    return {
                        'answer': top_match['answer'],
                        'original_question': top_match['question'],
                        'similarity_score': similarity_score,
                        'similar_questions': similar_questions.to_dict('records')
                    }
                else:
                    # If the similarity score is low, return a message indicating the chatbot only answers Sitar University related questions
                    return {
                        'answer': "I can only answer questions related to Sitare University. Please ask something specific about it.",
                        'similarity_score': similarity_score,
                        'similar_questions': similar_questions.to_dict('records')
                    }
            else:
                # If no similar questions are found
                return {
                    'answer': "Sorry, I can only answer questions related to Sitare University.",
                    'similar_questions': []
                }
        except Exception as e:
            print(f"Error getting answer: {e}")
            return {'answer': f"An error occurred: {e}"}

# Initialize Chatbot Instance
chatbot = CollegeChatbot(db_params)

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({'error': 'No query provided'}), 400
    
    result = chatbot.get_answer(user_query)
    
    response = {
        'question': user_query,
        'answer': result['answer']
    }
    
    if 'original_question' in result:
        response['original_question'] = result['original_question']
    
    if 'similar_questions' in result and result['similar_questions']:
        response['similar_questions'] = result['similar_questions']
    
    if 'similarity_score' in result:
        response['similarity_score'] = result['similarity_score']
    
    return jsonify(response)

# PostgreSQL Setup for Embeddings
try:
    connection = psycopg2.connect(
        host=db_params['host'],
        user=db_params['user'],
        password=db_params['password'],
        port=db_params['port'],
        database=db_params['database']
    )
    connection.autocommit = True
    cursor = connection.cursor()

    # Enable pgvector extension
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("pgvector extension is ready.")

    # Create table for storing embeddings
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS qa1_embeddings (
            id SERIAL PRIMARY KEY,
            question TEXT,
            answer TEXT,
            embedding vector(384)
        );
    """)
    print("Table 'qa1_embeddings' is ready.")

    # Close PostgreSQL connection
    cursor.close()
    connection.close()
    print("PostgreSQL connection is closed.")
except Exception as e:
    print(f"Error during PostgreSQL setup: {e}")

# Run Flask Application
if __name__ == '__main__':
    app.run(debug=True)
