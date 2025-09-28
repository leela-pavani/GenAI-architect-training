## Setup Instructions

### 1. Clone the Repository
git clone https://github.com/leela-pavani/GenAI-architect-training.git
cd your-repo

### 2. Create virtual environment 
python -m venv venv
venv\Scripts\activate

### 3. Install requirements
pip install -r requirements.txt

## 4. Add .env file with below parameters
AZURE_OPENAI_KEY 
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_DEPLOYMENT 
AZURE_OPENAI_VERSION 
AZURE_EMBEDDING_DEPLOYMENT
AZURE_CHAT_DEPLOYMENT

### 5. Run the respective notebook
python Assignment-2\assignment-2.py