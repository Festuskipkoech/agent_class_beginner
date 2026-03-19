# Setup Guide

## Install Required Modules

Run this command in your terminal:

```bash
pip install langchain-groq langchain-tavily langchain-core
```

---

## Get Your API Keys

### Groq API Key

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up with your email or Google account — no credit card needed
3. Once logged in, click **API Keys** on the left sidebar
4. Click **Create API Key**, give it any name, and copy it
5. It starts with `gsk_`

### Tavily API Key

1. Go to [https://app.tavily.com](https://app.tavily.com)
2. Sign up with your email — no credit card needed
3. Once logged in, your API key is shown on the dashboard
4. Copy it — it starts with `tvly-`

---

## Add Your Keys to the Code

Open `main.py` and replace the placeholder values with your actual keys:

```python
os.environ["TAVILY_API_KEY"] = "tvly-your-key-here"
os.environ["GROQ_API_KEY"]   = "gsk_your-key-here"
```

You are good to go. Run the code 