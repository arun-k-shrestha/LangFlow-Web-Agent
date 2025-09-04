# Web & Reddit Sentiment Agent

A simple agent that searches the **web** and **Reddit** to understand a company’s **sentiment** and **trends**. It uses the **OpenAI** API for analysis and the **Bright Data** API for search/snapshots.

### Quick install

# (optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

Note: Review requirements.txt to make sure the packages/versions match your code and avoid dependency conflicts.


### Environment

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
BRIGHTDATA_API_KEY=your_brightdata_key


This will:

1. query the web & Reddit via Bright Data, and 2) summarize sentiment/themes with OpenAI.

---

## Credits

I learned from a YouTube video and **credit code to Tech with Tim**.
