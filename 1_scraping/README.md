# 1_scraping

This folder contains scripts and data for scraping course and program information for the OpenCampus.edu website

---

##  Contents

- `scraping_pipeline.py` — Main script to scrape and preprocess data from web sources.
- `domain_data/` — Folder containing the raw and processed JSON data files for different domains (e.g., business, tech, degrees).

---

##  Libraries Used

- **Playwright** (`playwright.async_api`): For automated, headless web browsing and scraping.
- **asyncio**: For running asynchronous scraping tasks efficiently.
- **json**: For saving and loading structured data.

---

## How to Use

1. **Install dependencies (from project root):**
   ```bash
   pip install -r ../requirements.txt
   ```
2. **Run the scraping pipeline:**
   ```bash
   python scraping_pipeline.py
   ```
   This will populate the `domain_data/` folder with up-to-date course and program data in JSON format.

---

##  How it Works

1. **Fetches Web Pages:** Uses Playwright (with asyncio) to visit course and program pages on OpenCampus.edu.
2. **Extracts Data:** Selects and collects relevant information such as titles, descriptions, mentors, schedules, and more using CSS selectors.
3. **Cleans & Structures Data:** Processes and organizes the scraped data into structured JSON files for each domain.
4. **Saves Results:** Stores the processed data in the `domain_data/` folder, ready for use in vectorization and chatbot responses.

---

## 📄 Notes
- The output data will be used in later stages for vectorization and chatbot responses. 