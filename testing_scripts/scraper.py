import asyncio
from playwright.async_api import async_playwright
import json # Import the json module

# --- Your existing get_all_courses_by_domain function (no changes needed here) ---
async def get_all_courses_by_domain(url):
    """
    Launches a browser with Playwright, navigates to the URL, and extracts
    the list of course domains and the courses available under each.
    """
    all_domains_courses = {}
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True) 
            page = await browser.new_page()

            print(f"Navigating to {url}...")
            await page.goto(url, wait_until="networkidle") 
            await page.wait_for_load_state('domcontentloaded')

            h2_domain_selector = ".text-2xl.font-semibold.text-left.ml-3.md\\:ml-0"
            
            print(f"Waiting for domain H2 tags with selector: {h2_domain_selector}")
            await page.wait_for_selector(h2_domain_selector, state='visible', timeout=20000)

            h2_domain_locators = await page.locator(h2_domain_selector).all()

            if not h2_domain_locators:
                print(f"No domain h2 tags found on the page using selector: {h2_domain_selector}")
                return {}

            for h2_locator in h2_domain_locators:
                domain_name = await h2_locator.text_content()
                if not domain_name:
                    continue
                domain_name = domain_name.strip()
                print(f"\nProcessing domain: '{domain_name}'")
                
                current_domain_courses = []

                h2_element_handle = await h2_locator.element_handle()
                
                if h2_element_handle:
                    course_container_handle = await page.evaluate_handle('''h2 => {
                        let nextSibling = h2.nextElementSibling;
                        if (nextSibling && nextSibling.matches('.mt-2.mb-12')) {
                            return nextSibling;
                        }
                        return null;
                    }''', h2_element_handle)

                    if course_container_handle and await course_container_handle.evaluate('el => el !== null'):
                        courses_data_raw = await page.evaluate('''container => {
                            const courses = [];
                            const courseElements = container.querySelectorAll('.swiper-slide > a');
                            
                            courseElements.forEach(a_tag => {
                                const titleSpan = a_tag.querySelector('span.text-3xl.text-white');
                                if (titleSpan) {
                                    const title = titleSpan.textContent ? titleSpan.textContent.trim() : 'N/A';
                                    const url = a_tag.href;
                                    courses.push({ title, url });
                                }
                            });
                            return courses;
                        }''', course_container_handle)
                        
                        current_domain_courses.extend(courses_data_raw)
                        print(f"  Found {len(current_domain_courses)} courses under '{domain_name}'.")
                    else:
                        print(f"  No direct course container (div.mt-2.mb-12) found after '{domain_name}'.")
                
                all_domains_courses[domain_name] = current_domain_courses

            await browser.close()
            return all_domains_courses

    except Exception as e:
        print(f"An error occurred during scraping: {e}")
        print("Please ensure Playwright browsers are installed (`playwright install`) and your internet connection is stable.")
        return {}


# --- New functions for saving and loading data ---

def save_data_to_json(data, filename="opencampus_course_index.json"):
    """Saves the extracted course data to a JSON file."""
    try:
        # Before saving, add a 'details' key to each course dict as an empty dict
        # This makes the structure consistent for future updates
        for domain, courses in data.items():
            for course in courses:
                if 'details' not in course:
                    course['details'] = {}
                    
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"\nData successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving data to JSON: {e}")

def load_data_from_json(filename="opencampus_course_index.json"):
    """Loads course data from a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"\nData successfully loaded from {filename}")
        return data
    except FileNotFoundError:
        print(f"File '{filename}' not found. No data loaded.")
        return None
    except Exception as e:
        print(f"Error loading data from JSON: {e}")
        return None


if __name__ == "__main__":
    target_url = "https://edu.opencampus.sh/en"
    json_filename = "opencampus_course_index.json"

    # --- Option 1: Scrape and Save (Run this to initially populate the JSON) ---
    print("--- Starting initial data scraping and saving ---")
    all_extracted_data = asyncio.run(get_all_courses_by_domain(target_url))
    if all_extracted_data:
        save_data_to_json(all_extracted_data, json_filename)
    else:
        print("No data extracted to save.")
    
    # --- Option 2: Load Data (Run this in subsequent steps or to verify) ---
    print("\n--- Attempting to load data from JSON ---")
    loaded_data = load_data_from_json(json_filename)

    if loaded_data:
        print("\nSuccessfully loaded course data (first 2 courses of first domain):")
        # Print a sample to verify
        first_domain = next(iter(loaded_data)) # Get the name of the first domain
        print(f"Domain: {first_domain}")
        for i, course in enumerate(loaded_data[first_domain][:2]): # Print first 2 courses
            print(f"  - Title: {course['title']}")
            print(f"    URL: {course['url']}")
            print(f"    Details: {course['details']}") # Will be empty initially

        # In your next step (getting detailed course info):
        # You would iterate through `loaded_data`, access each course's 'url',
        # scrape that URL, and then update the 'details' dictionary for that course.
        # After updating, you'd save `loaded_data` back to the JSON file.

    else:
        print("No data loaded from JSON.")