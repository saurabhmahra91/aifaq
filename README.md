# Smart FAQ Assistant

This is a smart FAQ (Frequently Asked Questions) assistant that helps answer customer questions automatically.

## What Does This Do?

- **Smart Search**: Finds the best answers from your FAQ database, even when people ask questions differently
- **Learns Over Time**: Gets better at finding good answers based on user feedback (thumbs up/down)
- **Natural Responses**: Gives helpful, easy-to-understand answers
- **Easy to Use**: Simple web interface that anyone can use
- **Tracks Performance**: Shows you how well it's working with charts and statistics

## How It Works

1. Someone types a question
2. The system finds the most similar questions in your FAQ database
3. It creates a helpful answer based on your FAQ information
4. Users can give feedback to help it improve

## Quick Start - Step by Step Guide

### What You Need Before Starting

- A computer with internet connection
- An OpenAI account (you'll need to pay for API usage - usually costs just a few dollars)
- Basic familiarity with running commands on your computer

### Step 1: Get Your OpenAI API Key

1. Go to [OpenAI's website](https://platform.openai.com/api-keys)
2. Sign up or log in to your account
3. Create a new API key and copy it (you'll need this later)
4. Add some credit to your OpenAI account (usually $5-10 is plenty to start)

### Step 2: Download the Code

1. **Download the project files** to your computer
   - If you have git installed: open Terminal/Command Prompt and type:
     ```bash
     git clone <repository-url>
     cd aifaq
     ```
   - If you don't have git: download the ZIP file and extract it to a folder

### Step 3: Set Up Python Environment

1. **Open Terminal/Command Prompt** and navigate to the project folder:
   ```bash
   cd aifaq
   ```

2. **Create a virtual environment** (this keeps the project separate from other Python stuff):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On Mac/Linux:
     ```bash
     source venv/bin/activate
     ```
   
   You'll know it worked when you see `(venv)` at the beginning of your command line.

4. **Install the required software packages**:
   ```bash
   pip install .
   ```

### Step 4: Add Your OpenAI API Key

1. **Copy the example settings file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit the .env file** (use any text editor like Notepad):
   - Open the `.env` file
   - Replace `your-openai-api-key-here` with your actual OpenAI API key
   - Save the file

### Step 5: Start the Application

1. **Run the application**:
   ```bash
   streamlit run app.py
   ```

2. **Open your web browser** and go to: `http://localhost:8501`

3. **Start asking questions!** The system will find answers from your FAQ database.

### If Something Goes Wrong

- Make sure your virtual environment is activated (you should see `(venv)` in your command line)
- Check that your OpenAI API key is correct
- Make sure you have internet connection
- Ensure you have credit in your OpenAI account

## ⚙️ Basic Settings (Optional)

You can customize how the system works by editing the `.env` file:

### Main Settings

- **OPENAI_API_KEY**: Your OpenAI key (required - you got this in Step 1)
- **SIMILARITY_THRESHOLD**: How strict to be when matching questions (0.7 is good for most cases)
- **MAX_TOKENS**: Maximum length of responses (500 words is usually enough)

### Response Styles

You can choose how the system responds:
- **Helpful**: Friendly, conversational answers (recommended)
- **Concise**: Short, direct answers
- **Detailed**: Long, thorough explanations

## How Well Is It Working?

The system keeps track of how well it's doing:

- **Success Rate**: How often it finds good answers
- **User Feedback**: Thumbs up and thumbs down from users
- **Answer Quality**: How confident the system is in its answers
- **Learning Progress**: How the system improves over time

You can see these statistics in the web interface!

## What Can You Ask?

### Simple Questions
```
You ask: "What are your working hours?"
System says: "Our working hours are 9 AM to 5 PM, Monday through Friday."
```

### More Complex Questions
```
You ask: "How do I get help with my account?"
System finds: Multiple FAQs about password reset, customer support, etc.
System says: A helpful answer combining all the relevant information.
```

### Questions It Can't Answer
```
You ask: "What's the weather like?"
System says: "I don't have information about that. Please contact our support team for help."
```

## Dashboard Features

When you use the system, you'll see:

1. **Performance Charts**: Visual graphs showing how well the system is working
2. **Usage Stats**: Which questions are asked most often
3. **Feedback Summary**: How many thumbs up vs thumbs down
4. **System Health**: Whether everything is running smoothly

## Adding Your Own FAQs

To add new questions and answers:

1. **Open the `faqs.csv` file** in any spreadsheet program (like Excel) or text editor
2. **Add new rows** with your Question in column A and Answer in column B
3. **Save the file**
4. **Restart the application** - it will automatically learn the new FAQs

### Example FAQ Format
```
Question,Answer
"What are your hours?","We're open 9 AM to 5 PM, Monday through Friday."
"How do I reset my password?","Click 'Forgot Password' on the login page and follow the instructions."
```

## Testing Your System

After you get it running, try these things:

1. **Ask different types of questions:**
   - Questions that exactly match your FAQs
   - Questions that are similar but worded differently
   - Questions about topics not in your FAQ database

2. **Give feedback:**
   - Click thumbs up (👍) when you get good answers
   - Click thumbs down (👎) when answers aren't helpful
   - This helps the system learn!

3. **Check the dashboard:**
   - Look at the charts to see how well it's working
   - See which questions are asked most often

## Common Problems & Solutions

### "Can't connect to OpenAI"
- Check your internet connection
- Make sure your API key is correct in the `.env` file
- Verify you have credit in your OpenAI account

### "No answers found"
- Make sure your `faqs.csv` file has questions and answers
- Try asking simpler questions
- Check that the application started without errors

### "Poor quality answers"
- Add more specific FAQs to your `faqs.csv` file
- Give feedback with thumbs up/down to help it learn
- Try different response modes (Helpful, Concise, Detailed)

### Need Help?
If you're stuck, feel free to ask for help or report issues!

## License

This project is free to use under the MIT License - you can use it for personal or commercial projects.
