import asyncio
from playwright.async_api import async_playwright
import json
import time
import os # For creating directories

# --- Configuration ---
MAIN_URL = "https://edu.opencampus.sh/en"
COURSE_INDEX_FILENAME = "opencampus_course_index.json"
DOMAIN_DATA_DIR = "domain_data" # Directory to store domain-specific JSONs

# --- Utility Functions for JSON Saving/Loading ---
def save_data_to_json(data, filename):
    """Saves any Python dictionary/list to a JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"  Successfully saved data to {filename}")
    except Exception as e:
        print(f"  ERROR: Could not save data to {filename}: {e}")

def load_data_from_json(filename):
    """Loads data from a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"  Successfully loaded data from {filename}")
        return data
    except FileNotFoundError:
        print(f"  File '{filename}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"  ERROR: Could not decode JSON from '{filename}'. File might be corrupted.")
        return None
    except Exception as e:
        print(f"  ERROR: Could not load data from {filename}: {e}")
        return None

# --- Initial Scraping Function (for course_index.json) ---
async def get_all_courses_by_domain(url):
    """
    Scrapes the main page to get a list of courses categorized by domain.
    Returns a dictionary where keys are domain names and values are lists of courses.
    Each course is a dictionary with 'title' and 'url'.
    """
    all_domains_courses = {}
    print(f"\n[STEP 1/2: Initial Scrape] Starting initial scrape of main page: {url}")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await page.wait_for_load_state('domcontentloaded')

            h2_domain_selector = ".text-2xl.font-semibold.text-left.ml-3.md\\:ml-0"
            await page.wait_for_selector(h2_domain_selector, state='visible', timeout=20000)
            h2_domain_locators = await page.locator(h2_domain_selector).all()

            if not h2_domain_locators:
                print(f"  ERROR: No domain h2 tags found on the page using selector: {h2_domain_selector}")
                return {}

            for h2_locator in h2_domain_locators:
                domain_name = await h2_locator.text_content()
                if not domain_name:
                    print("  Skipping empty domain name.")
                    continue
                domain_name = domain_name.strip()
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
                                    courses.push({ title, url, details: {} }); # Add empty details dict for future use
                                }
                            });
                            return courses;
                        }''', course_container_handle)
                        current_domain_courses.extend(courses_data_raw)
                        print(f"  Found {len(current_domain_courses)} courses under '{domain_name}'.")
                    else:
                        print(f"  WARNING: No course container (div.mt-2.mb-12) found after '{domain_name}'.")
                all_domains_courses[domain_name] = current_domain_courses
            
            await browser.close()
            print("  Initial scrape of main page complete.")
            return all_domains_courses

    except Exception as e:
        print(f"  CRITICAL ERROR during initial scraping: {e}")
        return {}

# --- Detailed Scraping Function (for individual course pages) ---
async def get_course_details_from_page(page, course_url):
    """
    Navigates to a specific course URL and extracts detailed information.
    Includes robust error handling and debug statements only for missing fields.
    """
    course_details = {
        'url': course_url, # Store the URL with its details
        'description': None,
        'mentor': [],  # Initialize as a list for multiple mentors
        'what_we_will_learn': [],
        'day_time_summary': None,
        'ects': None,
        'location': None,
        'language': None,
        'information': None, # For combined "What's in it for you" and "Prior Knowledge"
        'course_sessions': [],
    }
    
    print(f"\n  Processing course page: {course_url}")

    try:
        await page.goto(course_url, wait_until="networkidle", timeout=30000)
        await page.wait_for_load_state('domcontentloaded')

        # --- 1. Course Description ---
        try:
            description_locator = page.locator(".flex.flex-1.flex-col.text-white.mb-4.lg\\:mb-20 span.text-2xl.mt-2").first
            if await description_locator.is_visible():
                course_details['description'] = (await description_locator.text_content()).strip()
            if not course_details['description']:
                print("    Description: Not found or empty.")
        except Exception as e:
            print(f"    ERROR: Could not get description: {e}")

        # --- 2. Course Mentor (Handles multiple mentors) ---
        try:
            # Select all mentor name spans within their specific div structure
            mentor_name_locators = await page.locator("div.mt-16.justify-start div.flex.items-start.flex.items-center.mb-6 div.flex.flex-col.text-lg span.mb-1").all()
            if mentor_name_locators:
                # Extract text content from each mentor span
                for locator in mentor_name_locators:
                    mentor_name = (await locator.text_content()).strip()
                    if mentor_name: # Only add if name is not empty
                        course_details['mentor'].append(mentor_name)
            if not course_details['mentor']:
                print("    Mentors: Not found or empty.")
        except Exception as e:
            print(f"    ERROR: Could not get mentor(s): {e}")

        # --- 3. What we will learn ---
        try:
            learn_elements = await page.locator("ul.list-disc.pb-12 li div.ml-2 span").all()
            if learn_elements:
                course_details['what_we_will_learn'] = [(await el.text_content()).strip() for el in learn_elements if await el.text_content()]
            if not course_details['what_we_will_learn']:
                print("    'What we will learn' section: Not found or no items visible.")
        except Exception as e:
            print(f"    ERROR: Could not get 'what we will learn': {e}")

        # --- 4. Course Day & Time (Summary) ---
        try:
            day_time_summary_locator = page.locator(".flex.flex-1.flex-col.text-white.mb-4.lg\\:mb-20 > span.text-xs").first
            if await day_time_summary_locator.is_visible():
                course_details['day_time_summary'] = (await day_time_summary_locator.text_content()).strip()
            if not course_details['day_time_summary']:
                print("    Day & Time Summary: Not found or empty.")
        except Exception as e:
            print(f"    ERROR: Could not get day & time summary: {e}")

        # --- 5. Course ECTS ---
        try:
            ects_label_locator = page.locator('span.text-lg.mt-2.text-center', has_text='ECTS').first
            if await ects_label_locator.is_visible():
                ects_value = await ects_label_locator.evaluate('''node => {
                    let current = node;
                    if (current && current.nextElementSibling) {
                        current = current.nextElementSibling;
                        if (current && current.nextElementSibling) {
                            return current.nextElementSibling.textContent.trim();
                        }
                    }
                    return null;
                }''')
                course_details['ects'] = ects_value if ects_value else None
            if not course_details['ects']:
                print("    ECTS: Not found or empty.")
        except Exception as e:
            print(f"    ERROR: Could not get ECTS: {e}")

        # --- 6. Location & Language taught ---
        # Location
        try:
            location_locator = page.locator('span.text-sm.mt-2.text-center', has_text='KIEL').first
            if await location_locator.is_visible():
                course_details['location'] = (await location_locator.text_content()).strip()
            else:
                online_location_locator = page.locator('span.text-sm.mt-2.text-center', has_text='ONLINE').first
                if await online_location_locator.is_visible():
                    course_details['location'] = (await online_location_locator.text_content()).strip()
            if not course_details['location']:
                print("    Location: Not found or empty.")
        except Exception as e:
            print(f"    ERROR: Could not get location: {e}")

        # Language
        try:
            language_locator = page.locator('span.text-sm.mt-2.text-center', has_text='ENGLISCH').first
            if await language_locator.is_visible():
                course_details['language'] = (await language_locator.text_content()).strip()
            else:
                german_language_locator = page.locator('span.text-sm.mt-2.text-center', has_text='DEUTSCH').first
                if await german_language_locator.is_visible():
                    course_details['language'] = (await german_language_locator.text_content()).strip()
            if not course_details['language']:
                print("    Language: Not found or empty.")
        except Exception as e:
            print(f"    ERROR: Could not get language: {e}")
        
        # --- 7. Combined Information Section (What's in It for You? & Prior Knowledge) ---
        try:
            combined_info_locator = page.locator(".flex.flex-wrap.bg-edu-course-invited.rounded-2xl.p-4.mx-6.xl\\:mx-0").first
            if await combined_info_locator.is_visible():
                full_text = await combined_info_locator.text_content()
                course_details['information'] = ' '.join(full_text.split()).strip()
            if not course_details['information']:
                print("    Combined Information section: Not found or empty.")
        except Exception as e:
            print(f"    ERROR: Could not get combined information section: {e}")
        
        # --- 8. Course Sessions ---
        try:
            session_list_items = await page.locator('ul.max-w-2xl li.flex.mb-4').all()
            if session_list_items:
                for item in session_list_items:
                    session = {}
                    # Date
                    date_span = item.locator('span.block.text-sm.sm\\:text-lg.font-semibold').first
                    session['date'] = (await date_span.text_content()).strip() if await date_span.is_visible() else None
                    
                    # Time
                    time_span = item.locator('span.text-sm.whitespace-nowrap').first
                    session['time'] = (await time_span.text_content()).strip() if await time_span.is_visible() else None

                    # Topic
                    topic_span = item.locator('span.block.text-sm.sm\\:text-lg.break-words').first
                    session['topic'] = (await topic_span.text_content()).strip() if await topic_span.is_visible() else None

                    # Location Text & URL
                    location_link = item.locator('span.text-sm.text-gray-400.ml-0.pl-0 a').first
                    if await location_link.is_visible():
                        session['location_text'] = (await location_link.text_content()).strip()
                        session['location_url'] = await location_link.get_attribute('href')
                    else:
                        location_span_text = item.locator('span.text-sm.text-gray-400.ml-0.pl-0').first
                        session['location_text'] = (await location_span_text.text_content()).strip() if await location_span_text.is_visible() else None
                        session['location_url'] = None
                    
                    course_details['course_sessions'].append(session)
            if not course_details['course_sessions']:
                print("    Course sessions: Not found or empty.")
        except Exception as e:
            print(f"    ERROR: Could not get course sessions: {e}")

    except Exception as e:
        print(f"  CRITICAL ERROR while visiting or parsing {course_url}: {e}")
    
    return course_details

# --- Main Execution Pipeline ---
async def run_scraping_pipeline():
    # 1. Ensure the output directory exists
    os.makedirs(DOMAIN_DATA_DIR, exist_ok=True)
    print(f"Ensured directory '{DOMAIN_DATA_DIR}' exists for domain-specific JSONs.")

    # 2. Load existing course index or perform initial scrape
    all_course_data = load_data_from_json(COURSE_INDEX_FILENAME)
    if all_course_data is None:
        all_course_data = await get_all_courses_by_domain(MAIN_URL)
        if all_course_data:
            save_data_to_json(all_course_data, COURSE_INDEX_FILENAME)
        else:
            print("Failed to get initial course index. Exiting pipeline.")
            return

    # 3. Iterate through courses and fetch detailed information
    print(f"\n[STEP 2/2: Detailed Scrape] Starting detailed course information scraping.")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True) # Set headless=False to see the browser
        page = await browser.new_page()

        total_domains = len(all_course_data)
        domains_processed = 0

        for domain_name, courses in all_course_data.items():
            domains_processed += 1
            print(f"\nProcessing detailed pages for domain: '{domain_name}' ({domains_processed}/{total_domains})")
            
            # Create a list to hold course data for the current domain
            current_domain_courses_list = []

            total_courses_in_domain = len(courses)
            courses_in_domain_processed = 0

            for course in courses:
                courses_in_domain_processed += 1
                print(f"  ({courses_in_domain_processed}/{total_courses_in_domain}) Processing '{course.get('title', 'Untitled Course')}'")
                
                # Check if 'details' key exists and is populated to skip re-scraping
                # Updated check to ensure the newly added/modified fields are present
                if (course.get('details') and 
                    course['details'].get('description') is not None and 
                    course['details'].get('mentor') is not None and # Check if mentor list exists (even if empty)
                    course['details'].get('what_we_will_learn') is not None and
                    course['details'].get('information') is not None and # Check for the combined info field
                    course['details'].get('course_sessions') is not None): # Check if sessions list exists (even if empty)
                    print(f"    Details for '{course.get('title', 'Untitled')}' already present and complete. Skipping.")
                    current_domain_courses_list.append(course) # Add to list for saving
                    continue

                detailed_info = await get_course_details_from_page(page, course['url'])
                course['details'] = detailed_info
                current_domain_courses_list.append(course) # Add to list for saving

                # Add a small delay to be polite to the server
                time.sleep(1) # Wait for 1 second between course pages

            # Save the current domain's data to a separate JSON file
            domain_filename = os.path.join(DOMAIN_DATA_DIR, f"{domain_name.replace(' ', '_').lower()}.json")
            save_data_to_json(current_domain_courses_list, domain_filename)
            print(f"  Finished processing domain '{domain_name}'.")

        await browser.close()
        print("\n[STEP 2/2: Detailed Scrape] Detailed scraping complete!")
    
    print("\n--- Pipeline finished. Check 'opencampus_course_index.json' and the 'domain_data' directory. ---")

if __name__ == "__main__":
    asyncio.run(run_scraping_pipeline())
